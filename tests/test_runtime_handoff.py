import asyncio
import hashlib
import json
import shutil
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

    async def snapshot():
        return {"ram_available_bytes": 1024**3}

    runtime._snapshot = snapshot
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


async def test_native_cancel_releases_slot_and_cleans_state(native):
    runtime, server = native
    started = asyncio.Event()
    handler = server.handle

    async def blocked(request):
        if request.url.path == "/completion":
            started.set()
            await asyncio.Event().wait()
        return await handler(request)

    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(blocked), base_url="http://native"
    )
    task = asyncio.create_task(runtime.prefill([1, 2, 3, 4]))
    await started.wait()
    await runtime.cancel()
    assert task.cancelled()
    assert not runtime._lock.locked()
    assert runtime._active is None
    assert server.prefix == []
    assert list(runtime._slots_dir.iterdir()) == []


async def test_split_preserves_canonical_tokens_and_reports_client_ttft(native):
    runtime, server = native
    # Simulate CUDA export using the same slot file format as the mocked Metal engine.
    exported = await runtime.prefill([1, 2, 3, 4])
    exported.manifest.source_backend = "cuda"
    exported.manifest.timings.update(
        {
            "peak_process_rss_bytes": 700 * 1024**2,
            "peak_gpu_used_bytes": 1024**3,
            "memory_sample_count": 2,
            "memory_sample_period_ms": 100,
        }
    )
    pipeline = Pipeline(runtime.settings)
    await pipeline.remote.close()
    pipeline.runtime = runtime
    pipeline.remote.profile = RuntimeProfile(gpu_layers=12, cpu_ffn=8)
    deleted = []
    transferred = exported.state_path.read_bytes()
    health = {
        "status": "ready",
        "profile": pipeline.remote.profile.model_dump(),
        "engine": {"effective_settings": {"effective_n_ctx": 8192, "total_slots": 1}},
    }

    async def worker(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/health":
            return httpx.Response(200, json=health)
        if request.url.path == "/prefill":
            assert body["token_ids"] == [1, 2, 3, 4]
            await asyncio.sleep(0.02)
            return httpx.Response(200, json=exported.manifest.model_dump(mode="json"))
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json={"deleted": True})
        assert request.url.path.endswith("/state")
        return httpx.Response(200, content=transferred)

    # The local downloaded artifact must be a different directory from the remote artifact.
    shutil.rmtree(exported.state_path.parent)
    pipeline.remote._client = httpx.AsyncClient(
        transport=httpx.MockTransport(worker), base_url="http://worker"
    )
    events = [
        event
        async for event in pipeline.generate(
            GenerationRequest(
                token_ids=[1, 2, 3, 4],
                mode="split",
                max_tokens=2,
            )
        )
    ]
    result = events[-1].result
    assert result.mode == "split"
    assert result.metrics.ttft_ms >= 20
    assert result.metrics.total_ms >= result.metrics.ttft_ms
    assert result.metrics.transfer_mbps > 0
    assert result.remote_profile.gpu_layers == 12
    assert result.remote_profile.cpu_ffn == 8
    assert result.effective_settings["remote"]["effective_n_ctx"] == 8192
    assert result.effective_settings["local"] == {}
    assert result.metrics.memory["remote_prefill_peak"]["gpu_used_max_bytes"] == 1024**3
    assert result.metrics.memory["remote_prefill_peak"]["sample_count"] == 2
    assert result.metrics.memory["before"]["ram_available_bytes"] == 1024**3
    assert deleted == [f"/artifacts/{exported.manifest.artifact_id}"]
    assert not exported.state_path.parent.exists()
    await pipeline.close()


async def test_empty_text_token_still_starts_client_ttft(native):
    runtime, server = native
    handler = server.handle

    async def empty_text(request):
        response = await handler(request)
        if request.url.path == "/completion" and json.loads(request.content).get("stream"):
            stream = response.text.replace('"content": "hello"', '"content": ""')
            return httpx.Response(200, text=stream)
        return response

    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(empty_text), base_url="http://native"
    )
    events = [event async for event in runtime.generate(GenerationRequest(token_ids=[1, 2]))]
    first = next(event for event in events if event.kind == "token")
    assert first.text == "" and first.token_id == 77
    assert events[-1].result.metrics.ttft_ms <= first.elapsed_ms


async def test_configuration_failure_restores_prior_runtime(native):
    runtime, _ = native
    previous = runtime.profile
    desired = previous.model_copy(update={"context_size": 16384})
    attempts = []

    async def close():
        runtime._client = None

    async def start():
        attempts.append(runtime.profile.context_size)
        if runtime.profile == desired:
            from pd_disaggregation_demo.runtime.engine import AdmissionError

            raise AdmissionError("Memory admission refused: reserve")
        runtime._client = httpx.AsyncClient(base_url="http://native")

    runtime.close = close
    runtime.start = start
    with pytest.raises(RuntimeError, match="Memory admission"):
        await runtime.configure(desired)
    assert runtime.profile == previous
    assert runtime._client is not None
    assert attempts == [16384, 8192]
    assert not runtime._lock.locked()
    await runtime._client.aclose()


async def test_remote_admission_error_has_explicit_type(tmp_path):
    from pd_disaggregation_demo.runtime.engine import AdmissionError

    remote = RemoteWorker(Settings(work_dir=tmp_path))
    await remote._client.aclose()
    remote._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                422,
                json={
                    "detail": {"code": "admission_refused", "message": "context exceeds reserve"},
                },
            )
        ),
        base_url="http://worker",
    )
    with pytest.raises(AdmissionError, match="context exceeds"):
        await remote._json("POST", "/configure", {})
    await remote.close()


async def test_closing_outer_stream_immediately_releases_native_slot(native):
    runtime, server = native
    pipeline = Pipeline(runtime.settings)
    pipeline.runtime = runtime
    iterator = pipeline.generate(GenerationRequest(token_ids=[1, 2, 3], max_tokens=2))
    while (await anext(iterator)).kind != "token":
        pass
    assert runtime._lock.locked()
    assert pipeline._lock.locked()
    await iterator.aclose()
    assert not runtime._lock.locked()
    assert not pipeline._lock.locked()
    assert server.prefix == []
    await pipeline.close()


async def test_native_chat_uses_model_apply_template(native):
    runtime, _ = native
    observed = []

    async def handler(request):
        observed.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"prompt": "native template prompt"})

    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://native"
    )
    messages = [{"role": "user", "content": "hello"}]
    assert await runtime.format_chat(messages) == "native template prompt"
    assert observed == [("/apply-template", {"messages": messages})]


async def test_export_disk_reserve_rechecked_before_native_work(native, monkeypatch):
    from types import SimpleNamespace

    from pd_disaggregation_demo.runtime.engine import AdmissionError

    runtime, server = native
    runtime._admission = SimpleNamespace(budgets={"state_bytes": 1024**3})
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=21 * 1024**3))
    with pytest.raises(AdmissionError, match="State disk admission"):
        await runtime.prefill([1, 2, 3, 4])
    assert server.calls == []


async def test_pipeline_session_owns_entire_sequence_and_is_reentrant(tmp_path):
    pipeline = Pipeline(Settings(work_dir=tmp_path))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def owner():
        async with pipeline.session():
            async with pipeline._request():
                assert pipeline._active is asyncio.current_task()
            assert pipeline._lock.locked()
            entered.set()
            await release.wait()

    task = asyncio.create_task(owner())
    await entered.wait()
    with pytest.raises(RuntimeError, match="one active"):
        async with pipeline.session():
            pass
    release.set()
    await task
    assert not pipeline._lock.locked()
    await pipeline.close()


async def test_memory_peak_monitor_retains_intermediate_maximum(native):
    runtime, _ = native
    rss = iter([10, 30, 20])

    async def snapshot():
        return {"process_rss_bytes": next(rss), "ram_available_bytes": 100}

    runtime._snapshot = snapshot
    runtime._peak_memory = {"sample_count": 0, "sample_period_ms": 100}
    await runtime._sample_peak()
    await runtime._sample_peak()
    await runtime._sample_peak()
    assert runtime._peak_memory["process_rss_max_bytes"] == 30
    assert runtime._peak_memory["sample_count"] == 3


async def test_same_profile_configure_restarts_exited_native_process(native):
    from types import SimpleNamespace

    runtime, _ = native
    runtime._process = SimpleNamespace(returncode=-9)
    restarted = []

    async def start():
        restarted.append(runtime.profile)
        runtime._process = SimpleNamespace(returncode=None)

    async def close():
        runtime._process = None
        if runtime._client:
            await runtime._client.aclose()
        runtime._client = None

    runtime.start = start
    runtime.close = close
    await runtime.configure(runtime.profile)
    assert restarted == [runtime.profile]


async def test_crashed_native_health_does_not_probe_replacement_listener(native):
    from types import SimpleNamespace

    runtime, server = native
    runtime._process = SimpleNamespace(returncode=-9)
    health = await runtime.health()
    assert health["status"] == "stopped"
    assert health["native_exit_code"] == -9
    assert server.calls == []
    runtime._process = None


async def test_exited_owned_process_refuses_requests_and_skips_erase(native):
    from types import SimpleNamespace

    runtime, server = native
    runtime._process = SimpleNamespace(returncode=-9)
    with pytest.raises(RuntimeError, match="Owned native llama-server exited"):
        await runtime.prefill([1, 2, 3, 4])
    assert server.calls == []
    assert not runtime._lock.locked()
    runtime._process = None


async def test_effective_props_accept_context_rounding_and_leave_unobserved_types_unknown(native):
    runtime, _ = native
    props = {
        "default_generation_settings": {"n_ctx": 8448},
        "total_slots": 1,
        "build_info": "b11552-23b0202a1",
        "model_ftype": "Q4_K - Medium",
    }
    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=props)),
        base_url="http://native",
    )
    await runtime._capture_native_props()
    assert runtime._effective_settings["effective_n_ctx"] == 8448
    assert runtime._effective_settings["cache_type_k"] is None
    assert runtime._effective_settings["effective_n_batch"] is None
    assert runtime._native_props["build_info"] == "b11552-23b0202a1"


@pytest.mark.parametrize("context,slots", [(8191, 1), (8192, 2)])
async def test_incompatible_effective_props_rejected(native, context, slots):
    runtime, _ = native
    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": context},
                    "total_slots": slots,
                },
            )
        ),
        base_url="http://native",
    )
    with pytest.raises(RuntimeError, match="Native"):
        await runtime._capture_native_props()


async def test_remote_effective_metadata_refreshed_only_after_configure(tmp_path):
    remote = RemoteWorker(Settings(work_dir=tmp_path))
    await remote._client.aclose()
    calls = []
    profile = RuntimeProfile(context_size=8192)
    current = profile.model_copy(update={"context_size": 4096})

    async def handler(request):
        nonlocal current
        calls.append((request.method, request.url.path))
        if request.url.path == "/configure":
            current = RuntimeProfile.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "profile": current.model_dump(),
                "engine": {"effective_settings": {"effective_n_ctx": current.context_size}},
            },
        )

    remote._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://worker"
    )
    await remote.configure_matching(profile)
    assert calls == [("GET", "/health"), ("POST", "/configure"), ("GET", "/health")]
    assert remote.effective_settings["effective_n_ctx"] == 8192
    calls.clear()
    await remote.configure_matching(profile)
    assert calls == [("GET", "/health")]
    await remote.close()
