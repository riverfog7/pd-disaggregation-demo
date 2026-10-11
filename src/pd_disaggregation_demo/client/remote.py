"""Worker control and bounded, verified native-state transport."""

import asyncio
import hashlib
import shutil
import time

import httpx

from ..artifacts import CHUNK_SIZE, write_manifest
from ..config.settings import Settings
from ..models import ArtifactManifest, PrefillArtifact, RuntimeFingerprint, RuntimeProfile


class RemoteWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.remote_url.rstrip("/"),
            timeout=settings.request_timeout,
            trust_env=False,
        )
        self.profile: RuntimeProfile | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _json(self, method: str, path: str, payload: dict | None = None) -> dict:
        response = await self._client.request(method, path, json=payload)
        if response.status_code == 422:
            detail = response.json().get("detail")
            if isinstance(detail, dict) and detail.get("code") == "admission_refused":
                from ..runtime.engine import AdmissionError

                raise AdmissionError(detail.get("message", "Remote memory admission refused"))
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict:
        return await self._json("GET", "/health")

    async def configure_matching(self, local: RuntimeProfile) -> RuntimeProfile:
        health = await self.health()
        current = RuntimeProfile.model_validate(health["profile"])
        # Retain CUDA worker placement, batching, and CPU thread choices.
        desired = current.model_copy(
            update={
                "model_id": local.model_id,
                "cache_type": local.cache_type,
                "context_size": local.context_size,
            }
        )
        if desired != current or health.get("status") != "ready":
            await self._json("POST", "/configure", desired.model_dump())
        self.profile = desired
        return desired

    async def cancel(self) -> None:
        await self._json("POST", "/cancel", {})

    async def delete(self, artifact_id: str) -> None:
        await self._json("DELETE", f"/artifacts/{artifact_id}")

    async def prefill(self, token_ids: list[int]) -> ArtifactManifest:
        result = await self._json("POST", "/prefill", {"token_ids": token_ids, "backend": "remote"})
        manifest = ArtifactManifest.model_validate(result)
        if manifest.token_ids != token_ids:
            raise ValueError("Remote prefill returned different canonical prompt tokens")
        if manifest.source_backend != "cuda":
            raise ValueError("Split prefill requires the CUDA worker")
        return manifest

    async def download(
        self, manifest: ArtifactManifest, fingerprint: RuntimeFingerprint | None = None
    ) -> tuple[PrefillArtifact, float]:
        if fingerprint is not None and manifest.fingerprint != fingerprint:
            raise ValueError("Worker artifact fingerprint does not match local runtime")
        # Never interpolate an unchecked server-supplied identifier into a filesystem path.
        if len(manifest.artifact_id) != 32 or any(
            c not in "0123456789abcdef" for c in manifest.artifact_id
        ):
            raise ValueError("Invalid artifact identifier")
        directory = self.settings.artifacts_dir / manifest.artifact_id
        directory.mkdir(parents=True, exist_ok=False)
        temporary = directory / "state.bin.part"
        start = time.perf_counter()
        try:
            digest = hashlib.sha256()
            size = 0
            async with self._client.stream(
                "GET", f"/artifacts/{manifest.artifact_id}/state"
            ) as response:
                response.raise_for_status()
                header = response.headers.get("content-length")
                if header is not None and int(header) != manifest.state_bytes:
                    raise ValueError("Remote state Content-Length mismatch")
                with temporary.open("wb") as target:
                    async for chunk in response.aiter_bytes(CHUNK_SIZE):
                        size += len(chunk)
                        if size > manifest.state_bytes:
                            raise ValueError("Remote state exceeded declared size")
                        digest.update(chunk)
                        await asyncio.to_thread(target.write, chunk)
            if size != manifest.state_bytes:
                raise ValueError("Remote state size mismatch")
            if digest.hexdigest() != manifest.state_sha256:
                raise ValueError("Remote state SHA256 mismatch")
            state = directory / "state.bin"
            temporary.replace(state)
            artifact = PrefillArtifact(manifest=manifest, state_path=state)
            write_manifest(artifact)
            return artifact, (time.perf_counter() - start) * 1000
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
