import asyncio
import hashlib
import sys
import types
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from pd_disaggregation_demo.backend.app import CHUNK_BYTES, _publish, create_worker_app
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import (
    ArtifactManifest,
    PrefillArtifact,
    RuntimeFingerprint,
)


class FakeRuntime:
    instances = []

    def __init__(self, settings, profile, backend):
        self.settings, self.profile, self.backend = settings, profile, backend
        self.fingerprint = RuntimeFingerprint(
            engine_revision=settings.engine_revision,
            model_sha256="a" * 64,
            cache_type_k="f16",
            cache_type_v="f16",
            context_size=profile.context_size,
        )
        self.started = self.closed = False
        self.block = False
        self.entered = asyncio.Event()
        self.cleaned = False
        FakeRuntime.instances.append(self)

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def health(self):
        return {"status": "ready"}

    async def configure(self, profile):
        self.profile = profile

    async def prefill(self, tokens):
        self.entered.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            identifier = uuid4().hex
            directory = self.settings.artifacts_dir / identifier
            directory.mkdir()
            payload = b"abc" * (CHUNK_BYTES // 3 + 10)
            path = directory / "state.bin"
            path.write_bytes(payload)
            manifest = ArtifactManifest(
                artifact_id=identifier,
                model_id=self.profile.model_id,
                fingerprint=self.fingerprint,
                token_ids=tokens,
                prefix_tokens=len(tokens) - 1,
                state_bytes=len(payload),
                state_sha256=hashlib.sha256(payload).hexdigest(),
                source_backend="cuda",
            )
            return PrefillArtifact(manifest=manifest, state_path=path)
        finally:
            self.cleaned = True


@pytest.fixture
async def worker(monkeypatch, tmp_path, request):
    module = types.ModuleType("pd_disaggregation_demo.runtime.engine")
    module.AdmissionError = type("AdmissionError", (RuntimeError,), {})
    module.LlamaRuntime = FakeRuntime
    monkeypatch.setitem(sys.modules, "pd_disaggregation_demo.runtime.engine", module)
    settings = Settings(
        work_dir=tmp_path, git_revision="f" * 40, artifact_ttl_seconds=getattr(request, "param", 60)
    )
    app = create_worker_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            yield client, app.state.runtime, settings
    assert app.state.runtime.closed


async def test_lifecycle_canonical_prefill_and_stream(worker):
    client, runtime, settings = worker
    assert runtime.started and runtime.backend == "cuda"
    health = (await client.get("/health")).json()
    assert health["git_revision"] == "f" * 40 and health["status"] == "ready"
    assert (await client.post("/prefill", json={"prompt": "abc"})).status_code == 422
    response = await client.post("/prefill", json={"token_ids": [1, 2, 3]})
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["prefix_tokens"] == 2 and manifest["expires_at"]
    identifier = manifest["artifact_id"]
    state = await client.get(f"/artifacts/{identifier}/state")
    assert len(state.content) > CHUNK_BYTES
    assert hashlib.sha256(state.content).hexdigest() == state.headers["x-state-sha256"]
    assert (await client.get(f"/artifacts/{identifier}/manifest")).json() == manifest
    assert (await client.delete(f"/artifacts/{identifier}")).status_code == 200
    assert not (settings.artifacts_dir / identifier).exists()


async def test_slot_busy_cancellation_and_reuse(worker):
    client, runtime, _ = worker
    runtime.block = True
    request = asyncio.create_task(client.post("/prefill", json={"token_ids": [1, 2]}))
    await runtime.entered.wait()
    assert (await client.post("/configure", json={})).status_code == 409
    assert (await client.post("/prefill", json={"token_ids": [1, 2]})).status_code == 409
    assert (await client.post("/cancel")).json()["cancelled"]
    assert (await request).status_code == 409
    assert runtime.cleaned
    runtime.block = False
    assert (await client.post("/prefill", json={"token_ids": [2, 3]})).status_code == 200


async def test_bad_uuids_symlinks_and_expiry(worker, tmp_path):
    client, _, settings = worker
    assert (await client.get("/artifacts/not-a-uuid/state")).status_code == 400
    identifier = uuid4().hex
    (settings.artifacts_dir / identifier).symlink_to(tmp_path.parent, target_is_directory=True)
    assert (await client.get(f"/artifacts/{identifier}/manifest")).status_code == 400
    manifest = (await client.post("/prefill", json={"token_ids": [1, 2]})).json()
    directory = settings.artifacts_dir / manifest["artifact_id"]
    parsed = ArtifactManifest.model_validate(manifest).model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    (directory / "manifest.json").write_text(parsed.model_dump_json())
    assert (await client.get(f"/artifacts/{directory.name}/state")).status_code == 410


async def test_corrupt_runtime_export_is_not_published(worker):
    _, runtime, settings = worker
    artifact = await runtime.prefill([1, 2])
    artifact.state_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size"):
        _publish(settings, artifact)
    assert not artifact.state_path.with_name("manifest.json").exists()


@pytest.mark.parametrize("worker", [1], indirect=True)
async def test_ttl_reaper_removes_only_owned_uuid_states(worker):
    client, _, settings = worker
    settings.artifact_ttl_seconds = 1
    unrelated = settings.artifacts_dir / "user-assets"
    unrelated.mkdir()
    response = await client.post("/prefill", json={"token_ids": [1, 2]})
    directory = settings.artifacts_dir / response.json()["artifact_id"]
    deadline = asyncio.get_running_loop().time() + 4
    while directory.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
    assert not directory.exists()
    assert unrelated.is_dir()


async def test_admission_has_machine_readable_error_code(worker, monkeypatch):
    client, runtime, _ = worker
    error_type = sys.modules["pd_disaggregation_demo.runtime.engine"].AdmissionError

    async def refused(profile):
        raise error_type("Not enough VRAM")

    monkeypatch.setattr(runtime, "configure", refused)
    response = await client.post("/configure", json={})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "admission_refused"
