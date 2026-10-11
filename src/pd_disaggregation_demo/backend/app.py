"""Single-slot CUDA worker; native states stay on disk during transfer."""

import asyncio
import hashlib
import shutil
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..config.settings import Settings
from ..models import ArtifactManifest, PrefillRequest, RuntimeProfile
from ..resources import capture_resources

CHUNK_BYTES = 1024 * 1024


def _artifact_directory(root: Path, artifact_id: str) -> Path:
    try:
        identifier = UUID(artifact_id)
    except (ValueError, AttributeError):
        raise HTTPException(400, "Invalid artifact UUID") from None
    if artifact_id not in (identifier.hex, str(identifier)):
        raise HTTPException(400, "Artifact UUID must be canonical")
    candidate = root / artifact_id
    if candidate.is_symlink() or candidate.resolve().parent != root.resolve():
        raise HTTPException(400, "Invalid artifact path")
    return candidate


def _read_manifest(directory: Path) -> ArtifactManifest:
    file = directory / "manifest.json"
    if file.is_symlink():
        raise HTTPException(400, "Invalid manifest path")
    try:
        manifest = ArtifactManifest.model_validate_json(file.read_text())
    except FileNotFoundError:
        raise HTTPException(404, "Artifact not found") from None
    except (ValueError, OSError):
        raise HTTPException(500, "Invalid artifact manifest") from None
    if manifest.artifact_id != directory.name:
        raise HTTPException(500, "Artifact UUID does not match its directory")
    if manifest.expires_at and manifest.expires_at <= datetime.now(UTC):
        raise HTTPException(410, "Artifact expired")
    return manifest


def _publish(settings: Settings, artifact) -> ArtifactManifest:
    manifest = artifact.manifest
    directory = _artifact_directory(settings.artifacts_dir, manifest.artifact_id)
    state = Path(artifact.state_path)
    if state.is_symlink() or state.resolve() != (directory / "state.bin").resolve():
        raise RuntimeError("Runtime state must live in its artifact directory")
    if not state.is_file() or state.stat().st_size != manifest.state_bytes:
        raise RuntimeError("Runtime state size does not match the manifest")
    digest = hashlib.sha256()
    with state.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest.state_sha256:
        raise RuntimeError("Runtime state checksum does not match the manifest")
    manifest = manifest.model_copy(
        update={"expires_at": datetime.now(UTC) + timedelta(seconds=settings.artifact_ttl_seconds)}
    )
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2))
    temporary.replace(directory / "manifest.json")
    return manifest


def create_worker_app(settings: Settings, profile: RuntimeProfile = RuntimeProfile()) -> FastAPI:
    # Late import lets resource/deployment tooling work without starting an engine.
    from ..runtime.engine import AdmissionError, LlamaRuntime

    slot = asyncio.Lock()
    active: asyncio.Task | None = None
    downloads: dict[str, int] = {}

    async def reap():
        while True:
            await asyncio.sleep(min(60, max(1, settings.artifact_ttl_seconds // 2)))
            for directory in settings.artifacts_dir.iterdir():
                if (
                    not directory.is_dir()
                    or directory.is_symlink()
                    or downloads.get(directory.name)
                ):
                    continue
                try:
                    _artifact_directory(settings.artifacts_dir, directory.name)
                    manifest = ArtifactManifest.model_validate_json(
                        (directory / "manifest.json").read_text()
                    )
                    expired = manifest.expires_at and manifest.expires_at <= datetime.now(UTC)
                except (HTTPException, OSError, ValueError):
                    # Only reap stale UUID directories left behind by failed exports.
                    try:
                        _artifact_directory(settings.artifacts_dir, directory.name)
                        expired = datetime.fromtimestamp(directory.stat().st_mtime, UTC) < (
                            datetime.now(UTC) - timedelta(seconds=settings.artifact_ttl_seconds)
                        )
                    except (HTTPException, OSError):
                        continue
                if expired and not slot.locked():
                    await asyncio.to_thread(shutil.rmtree, directory, True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        runtime = LlamaRuntime(settings, profile=profile, backend="cuda")
        app.state.runtime = runtime
        reaper = None
        try:
            await runtime.start()
            reaper = asyncio.create_task(reap())
            yield
        finally:
            if active and not active.done():
                active.cancel()
                with suppress(asyncio.CancelledError):
                    await active
            if reaper:
                reaper.cancel()
                with suppress(asyncio.CancelledError):
                    await reaper
            await runtime.close()

    app = FastAPI(title="PD CUDA worker", lifespan=lifespan)

    @app.get("/health")
    async def health():
        runtime = app.state.runtime
        engine_health = await runtime.health()
        if hasattr(engine_health, "model_dump"):
            engine_health = engine_health.model_dump(mode="json")
        fingerprint = runtime.fingerprint
        return {
            "status": engine_health.get("status", "unknown"),
            "engine": engine_health,
            "busy": slot.locked(),
            "profile": runtime.profile.model_dump(mode="json"),
            "fingerprint": fingerprint.model_dump(mode="json") if fingerprint else None,
            "git_revision": settings.git_revision,
            "resources": (await asyncio.to_thread(capture_resources, settings.work_dir)).model_dump(
                mode="json"
            ),
        }

    @app.get("/capabilities")
    async def capabilities():
        return {
            "backend": "cuda",
            "prefill": True,
            "state_format_version": 1,
            "cache_types": ["f16", "q8_0"],
            "max_concurrent_requests": 1,
            "engine_revision": settings.engine_revision,
            "git_revision": settings.git_revision,
        }

    async def run_in_slot(operation):
        nonlocal active
        if slot.locked():
            raise HTTPException(409, "Worker is busy")
        async with slot:
            active = asyncio.create_task(operation())
            try:
                return await active
            except asyncio.CancelledError:
                raise HTTPException(409, "Demo request cancelled") from None
            except HTTPException:
                raise
            except AdmissionError as error:
                raise HTTPException(
                    422, {"code": "admission_refused", "message": str(error)}
                ) from error
            except (ValueError, MemoryError) as error:
                raise HTTPException(422, str(error)) from error
            except Exception as error:
                raise HTTPException(503, str(error)) from error
            finally:
                active = None

    @app.post("/configure")
    async def configure(request: RuntimeProfile):
        async def operation():
            await app.state.runtime.configure(request)
            return {"profile": app.state.runtime.profile.model_dump(mode="json")}

        return await run_in_slot(operation)

    @app.post("/prefill", response_model=ArtifactManifest)
    async def prefill(request: PrefillRequest):
        if request.token_ids is None or len(request.token_ids) < 2:
            raise HTTPException(422, "Remote prefill requires full canonical token_ids with N >= 2")
        if any(token < 0 for token in request.token_ids):
            raise HTTPException(422, "Token IDs must be nonnegative")
        if len(request.token_ids) > app.state.runtime.profile.context_size:
            raise HTTPException(422, "Prompt exceeds configured context")

        async def operation():
            artifact = await app.state.runtime.prefill(request.token_ids)
            if artifact.manifest.token_ids != request.token_ids:
                raise RuntimeError("Runtime returned different prompt tokens")
            return await asyncio.to_thread(_publish, settings, artifact)

        return await run_in_slot(operation)

    @app.get("/artifacts/{artifact_id}/manifest", response_model=ArtifactManifest)
    async def manifest(artifact_id: str):
        directory = _artifact_directory(settings.artifacts_dir, artifact_id)
        return await asyncio.to_thread(_read_manifest, directory)

    @app.get("/artifacts/{artifact_id}/state")
    async def state(artifact_id: str):
        directory = _artifact_directory(settings.artifacts_dir, artifact_id)
        manifest = await asyncio.to_thread(_read_manifest, directory)
        state_path = directory / "state.bin"
        if state_path.is_symlink():
            raise HTTPException(400, "Invalid state path")
        try:
            stream = state_path.open("rb")
        except FileNotFoundError:
            raise HTTPException(404, "Artifact state not found") from None
        downloads[artifact_id] = downloads.get(artifact_id, 0) + 1

        async def chunks():
            try:
                while chunk := await asyncio.to_thread(stream.read, CHUNK_BYTES):
                    yield chunk
            finally:
                stream.close()
                downloads[artifact_id] -= 1

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(manifest.state_bytes),
                "X-State-SHA256": manifest.state_sha256,
            },
        )

    @app.delete("/artifacts/{artifact_id}")
    async def delete(artifact_id: str):
        directory = _artifact_directory(settings.artifacts_dir, artifact_id)
        if slot.locked() or downloads.get(artifact_id):
            raise HTTPException(409, "Artifact is in use")
        await asyncio.to_thread(shutil.rmtree, directory, True)
        return {"deleted": artifact_id}

    @app.post("/cancel")
    async def cancel():
        task = active
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, HTTPException):
                await task
        return {"cancelled": task is not None}

    return app
