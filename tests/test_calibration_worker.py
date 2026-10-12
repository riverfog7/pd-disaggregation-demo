import asyncio
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from pd_disaggregation_demo import calibration_worker as module
from pd_disaggregation_demo.backend import app as worker_module
from pd_disaggregation_demo.calibration import CPU_RECURRENT_PROJECTIONS, PressureError
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import ResourceSnapshot, RuntimeProfile
from pd_disaggregation_demo.runtime.memory_probe import parse_memory_estimate


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_requires_finite_deadlines(value, tmp_path):
    for key in ("timeout", "probe_timeout"):
        with pytest.raises(ValueError, match="finite and positive"):
            module.create_calibration_worker_app(
                Settings(work_dir=tmp_path), RuntimeProfile(), **{key: value}
            )


async def test_fixed_profile_refuses_reconfiguration_before_native_start(monkeypatch, tmp_path):
    profile = RuntimeProfile(context_size=33024, batch_size=32768, ubatch_size=12288)
    runtime = module.GuardedWorkerRuntime(
        Settings(work_dir=tmp_path),
        profile,
        "cuda",
        binary=Path("server"),
        model_path=Path("model"),
        estimate=parse_memory_estimate("Host 1 2 3\nCUDA0 4 5 6\n"),
        operation_timeout=1800,
    )
    calls = []

    async def start():
        calls.append("start")

    async def guard(target, operation, timeout):
        calls.append((target, timeout))
        return await operation

    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(module, "guarded", guard)
    with pytest.raises(ValueError, match="restart and probe again"):
        await runtime.configure(profile.model_copy(update={"context_size": 33280}))
    assert calls == []
    await runtime.configure(profile)
    assert calls == [(runtime, 1800), "start"]


async def test_prefill_is_interrupted_when_calibration_watchdog_refuses(monkeypatch, tmp_path):
    runtime = module.GuardedWorkerRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("server"),
        model_path=Path("model"),
        estimate=parse_memory_estimate("Host 1 2 3\nCUDA0 4 5 6\n"),
        operation_timeout=1,
    )
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def native_prefill(self, tokens):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def pressure():
        await entered.wait()
        raise PressureError("GPU free memory fell below 1 GiB")

    monkeypatch.setattr(module.CalibrationRuntime, "prefill", native_prefill)
    runtime.watchdog = SimpleNamespace(task=asyncio.create_task(pressure()))
    with pytest.raises(PressureError, match="1 GiB"):
        await runtime.prefill([1, 2])
    assert cleaned.is_set()


async def test_lifecycle_verifies_probes_and_owns_guard_without_resident_page_guard(
    monkeypatch, tmp_path
):
    calls = []
    model_path = tmp_path / "existing.gguf"
    model_path.write_bytes(b"existing")
    settings = Settings(work_dir=tmp_path, model_path=model_path)
    profile = RuntimeProfile(
        cache_type="q8_0", context_size=33024, batch_size=32768, ubatch_size=12288, cpu_ffn=64
    )
    estimate = parse_memory_estimate("Host 1 2 3\nCUDA0 4 5 6\n")

    def verify(verify_settings, model_id):
        calls.append(("verify", verify_settings.model_path, model_id))
        return model_path

    def probe(settings, requested, backend, path, timeout, *, tensor_overrides):
        calls.append(("probe", requested, backend, path, timeout, tensor_overrides))
        return estimate

    class Runtime:
        def __init__(self, settings, profile, backend, **kwargs):
            self.profile = profile
            self.fingerprint = None
            self.closed = False
            self.kwargs = kwargs
            calls.append(("runtime", kwargs))

        async def start(self):
            assert self.watchdog.task and not self.watchdog.task.done()
            calls.append("start")

        async def close(self):
            self.closed = True
            calls.append("close")

        async def health(self):
            return {"status": "ready", "calibration": {"fixed_profile": self.profile.model_dump()}}

        async def configure(self, requested):
            if requested != self.profile:
                raise ValueError("Profile needs a new probe")

    class Watchdog:
        def __init__(self, runtime, *, telemetry_path):
            runtime.watchdog = self
            self.task = None
            self.runtime = runtime
            calls.append(("watchdog", telemetry_path))

        async def begin(self):
            calls.append("begin")
            self.task = asyncio.create_task(asyncio.Event().wait())

        def mark_phase(self, phase):
            calls.append(("phase", phase))

        async def close(self):
            assert self.runtime.closed
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            calls.append("watchdog-close")

    monkeypatch.setattr(module, "validate_cgroup", lambda: tmp_path)
    monkeypatch.setattr(module, "ensure_model", verify)
    monkeypatch.setattr(module, "ensure_engine", lambda *_: Path("server"))
    monkeypatch.setattr(module, "probe_memory", probe)
    monkeypatch.setattr(module, "GuardedWorkerRuntime", Runtime)
    monkeypatch.setattr(module, "MemoryWatchdog", Watchdog)
    # Clean mmap cache can fill memory.current; ordinary worker guard must not run.
    monkeypatch.setattr(worker_module, "_cgroup_available", lambda: 0)
    monkeypatch.setattr(worker_module, "WATCHDOG_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(
        worker_module,
        "capture_resources",
        lambda _: ResourceSnapshot(
            platform="linux",
            cpu_count=4,
            ram_total_bytes=16 * 1024**3,
            ram_available_bytes=1024**3,
            gpu_free_bytes=0,
            disk_free_bytes=100 * 1024**3,
        ),
    )
    app = module.create_calibration_worker_app(
        settings,
        profile,
        cpu_recurrent_projections=True,
        timeout=1800,
        probe_timeout=90,
        telemetry_path=tmp_path / "samples.jsonl",
    )
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.02)
        assert not app.state.runtime.closed
        assert not hasattr(app.state, "resource_watchdog")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            health = (await client.get("/health")).json()
            assert health["status"] == "ready"
            assert health["engine"]["calibration"]["fixed_profile"] == profile.model_dump()
            assert (await client.post("/configure", json=profile.model_dump())).status_code == 200
            changed = profile.model_copy(update={"ubatch_size": 8192})
            assert (await client.post("/configure", json=changed.model_dump())).status_code == 422
    assert calls[0] == ("verify", model_path, profile.model_id)
    assert calls[1] == ("probe", profile, "cuda", model_path, 90, CPU_RECURRENT_PROJECTIONS)
    assert calls[-2:] == ["close", "watchdog-close"]
    assert app.state.runtime.kwargs["estimate"] is estimate
    assert app.state.runtime.kwargs["cgroup"] == tmp_path


async def test_existing_model_required_before_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "validate_cgroup", lambda: tmp_path)
    settings = Settings(work_dir=tmp_path, model_path=tmp_path / "absent.gguf")
    app = module.create_calibration_worker_app(settings, RuntimeProfile())
    with pytest.raises(FileNotFoundError, match="already-downloaded"):
        async with app.router.lifespan_context(app):
            pytest.fail("Missing weights must not start a worker")
