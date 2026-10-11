import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from pd_disaggregation_demo.artifacts import load_artifact, validate_artifact
from pd_disaggregation_demo.client.remote import RemoteWorker
from pd_disaggregation_demo.config.settings import ENGINE_REVISION, Settings
from pd_disaggregation_demo.models import (
    ArtifactManifest,
    DecodeRequest,
    GenerationRequest,
    RuntimeFingerprint,
    RuntimeProfile,
)
from pd_disaggregation_demo.pipeline import Pipeline
from pd_disaggregation_demo.runtime.engine import LlamaRuntime


def fingerprint():
    return RuntimeFingerprint(
        engine_revision=ENGINE_REVISION,
        model_sha256="a" * 64,
        cache_type_k="f16",
        cache_type_v="f16",
        context_size=8192,
    )


class NativeServer:
    def __init__(self, runtime):
        self.runtime = runtime
        self.prefix = []
        self.calls = []
        self.wrong_saved = False
        self.wrong_restored = False
        self.wrong_cache = False
        self.fail_stream = False

    async def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.path, request.url.query.decode(), body))
        action = request.url.params.get("action")
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})
        if action == "erase":
            self.prefix = []
            return httpx.Response(200, json={"n_erased": 0})
        if action == "save":
            path = self.runtime._slots_dir / body["filename"]
            path.write_bytes(json.dumps(self.prefix).encode())
            return httpx.Response(
                200,
                json={
                    "n_saved": len(self.prefix) + self.wrong_saved,
                    "timings": {"save_ms": 2},
                },
            )
        if action == "restore":
            self.prefix = json.loads((self.runtime._slots_dir / body["filename"]).read_bytes())
            return httpx.Response(
                200,
                json={
                    "n_restored": len(self.prefix) + self.wrong_restored,
                    "timings": {"restore_ms": 3},
                },
            )
        assert request.url.path == "/completion"
        if not body["stream"]:
            self.prefix = body["prompt"]
            assert body["n_predict"] == 1
            return httpx.Response(
                200,
                json={
                    "tokens": [99],
                    "content": "discard me",
                    "timings": {
                        "prompt_n": len(self.prefix),
                        "cache_n": 0,
                        "prompt_ms": 10,
                        "prompt_per_second": 300,
                    },
                },
            )
        cache_n = len(self.prefix) if body["cache_prompt"] else 0
        timing = {
            "cache_n": cache_n - self.wrong_cache,
            "prompt_n": len(body["prompt"]) - cache_n,
            "prompt_ms": 1 if cache_n else 10,
            "prompt_per_second": 1000,
            "predicted_n": 2,
            "predicted_ms": 5,
            "predicted_per_second": 400,
        }
        records = [
            {"content": "hello", "tokens": [77], "stop": False, "timings": timing},
            {"content": " world", "tokens": [78], "stop": False, "timings": timing},
        ]
        if not self.fail_stream:
            records.append({"content": "", "tokens": [], "stop": True, "timings": timing})
        stream = "".join(f"data: {json.dumps(record)}\n\n" for record in records)
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})


@pytest.fixture
async def native(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", work_dir=tmp_path / "work")
    runtime = LlamaRuntime(settings, backend="metal")
    runtime.fingerprint = fingerprint()
    runtime._slots_dir.mkdir(parents=True)
    server = NativeServer(runtime)
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handle), base_url="http://native"
    )
    yield runtime, server
    await runtime.close()


async def test_native_handoff_keeps_exact_n_minus_one_boundary(native):
    runtime, server = native
    tokens = [1, 2, 3, 4]
    artifact = await runtime.prefill(tokens)
    assert artifact.manifest.prefix_tokens == 3
    assert json.loads(artifact.state_path.read_bytes()) == tokens[:-1]
    assert load_artifact(artifact.state_path.parent) == artifact
    events = [event async for event in runtime.decode(artifact, DecodeRequest(max_tokens=2))]
    result = events[-1].result
    assert result.text == "hello world"
    assert result.metrics.cache_n == 3
    assert result.metrics.prompt_n == 1
    assert result.metrics.decode_tps == 400
    assert result.metrics.final_prompt_ms == 1
    assert result.metrics.ttft_ms >= result.metrics.restore_ms
    payloads = [body for path, _, body in server.calls if path == "/completion"]
    assert payloads[0]["prompt"] == tokens[:-1]
    assert payloads[1]["prompt"] == tokens
    assert payloads[1]["cache_prompt"] is True
    assert server.prefix == []
    assert list(runtime._slots_dir.iterdir()) == []


@pytest.mark.parametrize(
    "flag,match",
    [
        ("wrong_saved", "saved wrong token"),
        ("wrong_restored", "restored wrong token"),
        ("wrong_cache", "State handoff failed"),
    ],
)
async def test_boundary_failure_never_falls_back(native, flag, match):
    runtime, server = native
    if flag == "wrong_saved":
        server.wrong_saved = True
        with pytest.raises(RuntimeError, match=match):
            await runtime.prefill([1, 2, 3, 4])
        assert not list(runtime.settings.artifacts_dir.glob("*/state.bin"))
    else:
        artifact = await runtime.prefill([1, 2, 3, 4])
        setattr(server, flag, True)
        with pytest.raises(RuntimeError, match=match):
            _ = [event async for event in runtime.decode(artifact, DecodeRequest())]
    assert not runtime._lock.locked()
    assert server.prefix == []


async def test_local_starts_fresh_and_requires_terminal(native):
    runtime, server = native
    request = GenerationRequest(token_ids=[1, 2, 3, 4], max_tokens=2)
    result = [event async for event in runtime.generate(request)][-1].result
    assert result.metrics.cache_n == 0
    assert result.metrics.prompt_n == 4
    assert result.metrics.prefill_tps == 1000
    server.fail_stream = True
    with pytest.raises(RuntimeError, match="without a terminal"):
        _ = [event async for event in runtime.generate(request)]
    assert not runtime._lock.locked()


async def test_corrupt_state_rejected_before_restore(native):
    runtime, server = native
    artifact = await runtime.prefill([1, 2, 3, 4])
    artifact.state_path.write_bytes(b"x" * artifact.manifest.state_bytes)
    with pytest.raises(ValueError, match="SHA256"):
        validate_artifact(artifact)
    with pytest.raises(ValueError, match="SHA256"):
        _ = [event async for event in runtime.decode(artifact, DecodeRequest())]
    assert not any("restore" in query for _, query, _ in server.calls)


def test_launch_flags_enforce_explicit_configuration(tmp_path):
    runtime = LlamaRuntime(Settings(work_dir=tmp_path), RuntimeProfile(cpu_ffn=3), backend="metal")
    args = runtime._arguments(Path("server"), Path("model"))
    for flag, value in [
        ("--cache-ram", "0"),
        ("--ctx-checkpoints", "0"),
        ("--flash-attn", "on"),
        ("--fit", "off"),
        ("--spec-type", "none"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--no-repack" in args and "--no-host" in args


async def test_remote_download_bounds_and_checksum(tmp_path):
    data = b"native recurrent and kv state"
    settings = Settings(work_dir=tmp_path)
    remote = RemoteWorker(settings)
    await remote._client.aclose()
    manifest = ArtifactManifest(
        model_id="qwen3.5-4b",
        fingerprint=fingerprint(),
        token_ids=[1, 2],
        prefix_tokens=1,
        source_backend="cuda",
        state_bytes=len(data),
        state_sha256=hashlib.sha256(data).hexdigest(),
    )
    remote._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data)),
        base_url="http://worker",
    )
    artifact, duration = await remote.download(manifest, fingerprint())
    assert artifact.state_path.read_bytes() == data
    assert duration >= 0
    await remote.close()
    broken = manifest.model_copy(update={"artifact_id": "b" * 32, "state_sha256": "c" * 64})
    remote = RemoteWorker(settings)
    await remote._client.aclose()
    remote._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data)),
        base_url="http://worker",
    )
    with pytest.raises(ValueError, match="SHA256"):
        await remote.download(broken, fingerprint())
    assert not (settings.artifacts_dir / broken.artifact_id).exists()
    await remote.close()


async def test_pipeline_cancel_waits_for_slot_cleanup(tmp_path):
    pipeline = Pipeline(Settings(work_dir=tmp_path))
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def generation(request):
        try:
            started.set()
            await asyncio.Event().wait()
            yield None
        finally:
            cleaned.set()

    pipeline.runtime.generate = generation

    async def consume():
        return [event async for event in pipeline.generate(GenerationRequest())]

    task = asyncio.create_task(consume())
    await started.wait()
    with pytest.raises(RuntimeError, match="one active"):
        _ = [event async for event in pipeline.generate(GenerationRequest())]
    await pipeline.cancel()
    assert task.cancelled()
    assert cleaned.is_set()
    assert not pipeline._lock.locked()
    await pipeline.close()
