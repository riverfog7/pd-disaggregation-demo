"""Meaningful guards and cleanup checks without running a native GPU workload."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pd_disaggregation_demo import calibration as c
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import ResourceSnapshot, RuntimeProfile
from pd_disaggregation_demo.runtime.memory_probe import parse_memory_estimate


def resources(**overrides):
    return {
        "ram_available_bytes": 64 * c.GiB,
        "gpu_free_bytes": 8 * c.GiB,
        "metal_recommended_bytes": 48 * c.GiB,
        **overrides,
    }


def group(**overrides):
    return {
        "limit_bytes": 8 * c.GiB,
        "swap_limit_bytes": 0,
        "nonreclaimable_headroom_bytes": 7 * c.GiB,
        **overrides,
    }


def estimate():
    return parse_memory_estimate("Host 16000 100 100\nCUDA0 1000 100 100\n")


MMAP = ["--no-host", "--no-repack", "--load-mode", "mmap"]


def test_cuda_mmap_weights_can_exceed_cgroup_but_nonreclaimable_cannot():
    result = c.preflight(estimate(), resources(), "cuda", MMAP, group())
    assert result["allowed"]
    assert result["budgets"]["native_host_total_bytes"] > 8 * c.GiB
    rejected = c.preflight(
        estimate(), resources(), "cuda", MMAP, group(nonreclaimable_headroom_bytes=c.GiB)
    )
    assert not rejected["allowed"]
    assert any("Nonreclaimable" in reason for reason in rejected["reasons"])


@pytest.mark.parametrize("arguments", [[], ["--no-host", "--no-repack", "--load-mode", "buffer"]])
def test_reclaimable_exception_requires_all_native_flags(arguments):
    assert not c.preflight(estimate(), resources(), "cuda", arguments, group())["allowed"]


@pytest.mark.parametrize(
    "snapshot,cgroup",
    [
        (resources(gpu_free_bytes=2 * c.GiB), group()),
        (resources(gpu_free_bytes=None), group()),
        (resources(ram_available_bytes=2 * c.GiB), group()),
        (resources(), group(swap_limit_bytes=None)),
        (resources(), group(limit_bytes=None)),
    ],
)
def test_cuda_preflight_refuses_missing_capacity_and_swap(snapshot, cgroup):
    assert not c.preflight(estimate(), snapshot, "cuda", MMAP, cgroup)["allowed"]


def test_metal_conservatively_counts_both_host_and_device():
    estimated = parse_memory_estimate("Host 1000 100 100\nMTL0 1000 100 100\n")
    assert c.preflight(estimated, resources(), "metal", MMAP, None)["allowed"]
    rejected = c.preflight(estimated, resources(ram_available_bytes=8 * c.GiB), "metal", MMAP, None)
    assert not rejected["allowed"]
    assert "unified_required_with_reserve_bytes" in rejected["budgets"]


def write_group(path, *, maximum=8 * c.GiB, current=7 * c.GiB, file=6 * c.GiB, anon=c.GiB, dirty=0):
    path.mkdir(exist_ok=True)
    (path / "memory.max").write_text(str(maximum))
    (path / "memory.current").write_text(str(current))
    (path / "memory.swap.max").write_text("0")
    (path / "memory.stat").write_text(
        f"anon {anon}\nfile {file}\nfile_dirty {dirty}\nfile_writeback 0\nshmem 0\n"
    )
    (path / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")


def test_clean_mmap_cache_near_cap_does_not_trigger_guard(tmp_path):
    write_group(tmp_path)
    sample = c.cgroup_sample(tmp_path)
    assert sample["nonreclaimable_headroom_bytes"] == 7 * c.GiB
    assert not c.pressure_reasons({**resources(), "cgroup": sample}, "cuda")
    write_group(tmp_path, current=8 * c.GiB - 128 * c.MiB, file=0)
    assert c.pressure_reasons({**resources(), "cgroup": c.cgroup_sample(tmp_path)}, "cuda")


def test_dirty_file_is_not_reclaimable_and_ancestor_usage_counts(tmp_path):
    write_group(tmp_path, current=7 * c.GiB, file=6 * c.GiB, dirty=2 * c.GiB)
    child = tmp_path / "child"
    write_group(child, maximum=4 * c.GiB, current=c.GiB, file=0)
    sample = c.cgroup_sample(child)
    assert sample["nonreclaimable_headroom_bytes"] == 3 * c.GiB
    write_group(tmp_path, current=8 * c.GiB - 64 * c.MiB, file=0)
    sample = c.cgroup_sample(child)
    assert sample["nonreclaimable_headroom_bytes"] == 64 * c.MiB


def test_fake_or_unbounded_cgroup_is_rejected(tmp_path, monkeypatch):
    write_group(tmp_path)
    monkeypatch.setattr(c, "current_cgroup", lambda: tmp_path)
    with pytest.raises(ValueError, match="invoking"):
        c.validate_cgroup(tmp_path / "fake")
    (tmp_path / "memory.max").write_text("max")
    with pytest.raises(ValueError, match="finite"):
        c.validate_cgroup()


def test_native_effective_settings_parser_reports_actual_and_unknown():
    value = c.effective_from_log(
        "n_ctx = 8448\nn_batch = 8192\nn_ubatch = 512\nflash_attn = enabled\n"
    )
    assert value["effective_n_ctx"] == 8448
    assert value["effective_n_ubatch"] == 512
    assert value["flash_attention"] is True
    assert c.effective_from_log("unrelated")["flash_attention"] is None
    assert c.effective_from_log("flash_attn = disabled")["flash_attention"] is False


class Owned:
    returncode = None
    pid = 100

    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


async def test_deadline_terminates_only_owned_process_and_cancels_operation():
    owned = Owned()
    monitor = asyncio.create_task(asyncio.Event().wait())
    runtime = SimpleNamespace(_process=owned, watchdog=SimpleNamespace(task=monitor))
    cancelled = asyncio.Event()

    async def operation():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        with pytest.raises(TimeoutError):
            await c.guarded(runtime, operation(), 0.01)
        assert owned.terminated and not owned.killed
        assert cancelled.is_set()
        assert not monitor.done()
    finally:
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor


async def test_live_pressure_interrupts_inflight_loading(tmp_path):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    runtime._process = Owned()
    watchdog = c.MemoryWatchdog(runtime)
    watchdog.sample = lambda: {**resources(gpu_free_bytes=512 * c.MiB), "cgroup": group()}
    watchdog.task = asyncio.create_task(watchdog.run())
    try:
        with pytest.raises(c.PressureError, match="GPU"):
            await c.guarded(runtime, asyncio.Event().wait(), 1)
        assert runtime._process.terminated
        assert watchdog.summary["pressure"]
    finally:
        await watchdog.close()


async def test_runtime_requires_guard_before_launch(tmp_path):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "metal",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    with pytest.raises(RuntimeError, match="active MemoryWatchdog"):
        await runtime.start()


def test_partial_report_atomic_and_invalid_report_leaves_old_file(tmp_path):
    path = tmp_path / "report.json"
    c.atomic_report(path, {"status": "running", "candidates": []})
    with pytest.raises(ValueError):
        c.atomic_report(path, {"bad": float("nan")})
    assert json.loads(path.read_text())["status"] == "running"
    assert list(tmp_path.iterdir()) == [path]


async def test_probe_only_verifies_model_once_and_never_launches(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.touch()
    settings = Settings(model_path=model_path, work_dir=tmp_path)
    verified = []

    def verify(settings, model_id):
        verified.append((settings.model_path, model_id))
        return model_path

    monkeypatch.setattr(c, "ensure_model", verify)
    monkeypatch.setattr(
        c, "probe_memory", lambda *a, **kw: parse_memory_estimate("Host 100 1 1\nMTL0 100 1 1\n")
    )
    monkeypatch.setattr(
        c,
        "capture_resources",
        lambda _: ResourceSnapshot(
            platform="darwin",
            cpu_count=8,
            ram_total_bytes=64 * c.GiB,
            ram_available_bytes=32 * c.GiB,
            disk_free_bytes=100 * c.GiB,
            metal_recommended_bytes=48 * c.GiB,
        ),
    )
    monkeypatch.setattr(c, "ensure_engine", lambda *a: pytest.fail("probe must not launch server"))
    report = await c.calibrate(
        settings,
        backend="metal",
        output=tmp_path / "report.json",
        ubatches=[128, 256],
        probe_only=True,
        progress=lambda _: None,
    )
    assert report["status"] == "completed"
    assert len(verified) == 1
    assert report["measured_safe_ubatches"] == []
    assert [item["status"] for item in report["candidates"]] == [
        "estimated_safe_unmeasured",
        "estimated_safe_unmeasured",
    ]
    assert report["requested_profile"]["context_size"] == 8320
    assert report["requested_profile"]["batch_size"] == 8192


async def test_missing_model_cannot_start_download(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ensure_model", lambda *a: pytest.fail("no download allowed"))
    report = await c.calibrate(
        Settings(work_dir=tmp_path, cache_dir=tmp_path),
        backend="metal",
        output=tmp_path / "report.json",
        probe_only=True,
        progress=lambda _: None,
    )
    assert report["status"] == "stopped"
    assert "already-downloaded" in report["error"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"flash_attention": None},
        {"flash_attention": False},
        {"effective_n_ubatch": 256},
        {"effective_n_batch": None},
        {"effective_n_ctx": 128},
    ],
)
def test_effective_native_flags_required_for_measurement(overrides):
    observed = {
        "effective_n_ctx": 8448,
        "effective_n_batch": 8192,
        "effective_n_ubatch": 128,
        "flash_attention": True,
    }
    profile = RuntimeProfile(context_size=8320, batch_size=8192, ubatch_size=128)
    c.validate_effective(observed, profile)
    with pytest.raises(RuntimeError):
        c.validate_effective({**observed, **overrides}, profile)


async def test_any_cgroup_oom_event_stops_escalation(tmp_path):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    runtime._process = Owned()
    watchdog = c.MemoryWatchdog(runtime)
    watchdog.events_before = {"oom": 0, "oom_kill": 0}
    watchdog.sample = lambda: {**resources(), "cgroup": group(events={"oom": 1, "oom_kill": 0})}
    with pytest.raises(c.PressureError, match="OOM"):
        await watchdog.run()
    assert runtime._process.terminated


async def test_runtime_estimate_cannot_be_reused_after_profile_change(tmp_path):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "metal",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    runtime.profile = runtime.profile.model_copy(update={"ubatch_size": 256})
    with pytest.raises(RuntimeError, match="original profile"):
        await runtime.start()


def test_final_sample_cannot_certify_a_violated_reserve(tmp_path):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    watchdog = c.MemoryWatchdog(runtime)
    safe = {**resources(), "cgroup": group(events={"oom": 0})}
    watchdog.check_sample(safe)
    with pytest.raises(c.PressureError, match="GPU"):
        watchdog.check_sample({**safe, "gpu_free_bytes": 512 * c.MiB})
    assert watchdog.summary["gpu_free_min_bytes"] == 512 * c.MiB
    with pytest.raises(c.PressureError, match="OOM"):
        watchdog.check_sample({**safe, "cgroup": group(events={"oom": 1})})


def test_calibration_projection_placement_matches_estimator_and_keeps_worker_clean(
    tmp_path, monkeypatch
):
    import subprocess

    from pd_disaggregation_demo.runtime import memory_probe
    from pd_disaggregation_demo.runtime.engine import LlamaRuntime

    profile = RuntimeProfile(
        model_id="qwen3.5-27b",
        cache_type="q8_0",
        context_size=8320,
        batch_size=8192,
        ubatch_size=512,
        cpu_ffn=64,
    )
    model = tmp_path / "model"
    settings = Settings(work_dir=tmp_path)
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "Host 14000 1 1\nCUDA0 1941 430 525\n", "")

    monkeypatch.setattr(memory_probe.subprocess, "run", fake_run)
    estimated = memory_probe.estimate_memory(
        Path("helper"), profile, "cuda", model, tensor_overrides=c.CPU_RECURRENT_PROJECTIONS
    )
    runtime = c.CalibrationRuntime(
        settings,
        profile,
        "cuda",
        binary=Path("server"),
        model_path=model,
        estimate=estimated,
        tensor_overrides=c.CPU_RECURRENT_PROJECTIONS,
    )
    native = runtime._arguments(Path("server"), model)
    override = c.CPU_RECURRENT_PROJECTIONS[0]
    assert calls[0][calls[0].index("--override-tensor") + 1] == override
    assert native[native.index("--override-tensor") + 1] == override
    assert native[native.index("--n-cpu-ffn") + 1] == "64"
    assert native[native.index("--flash-attn") + 1] == "on"
    assert "--op-offload" in native
    assert "--override-tensor" not in LlamaRuntime(settings, profile, "cuda")._arguments(
        Path("server"), model
    )
    assert list(estimated.arguments) == calls[0]


def test_optional_native_process_metrics_access_denied_does_not_abort_guards(tmp_path, monkeypatch):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    runtime._process = Owned()
    watchdog = c.MemoryWatchdog(runtime)
    snapshot = ResourceSnapshot(
        platform="linux",
        cpu_count=4,
        ram_total_bytes=32 * c.GiB,
        ram_available_bytes=16 * c.GiB,
        disk_free_bytes=100 * c.GiB,
        gpu_free_bytes=4 * c.GiB,
    )
    monkeypatch.setattr(c, "capture_resources", lambda _: snapshot)

    def denied(pid):
        raise c.psutil.AccessDenied(pid)

    monkeypatch.setattr(c.psutil, "Process", denied)
    watchdog.gpu.sample = lambda: {"gpu_free_bytes": 4 * c.GiB}
    sample = watchdog.sample()
    assert "AccessDenied" in sample["native_metrics_unavailable"]
    assert sample["gpu_free_bytes"] == 4 * c.GiB
    native = runtime._arguments(Path("server"), Path("model"))
    assert native[native.index("--verbosity") + 1] == "4"


def test_cgroup_pressure_stalls_are_observed_and_missing_files_optional(tmp_path):
    write_group(tmp_path)
    (tmp_path / "memory.pressure").write_text(
        "some avg10=0.5 avg60=0.1 avg300=0.0 total=123456\n"
        "full avg10=0.2 avg60=0.0 avg300=0.0 total=54321\n"
    )
    (tmp_path / "io.pressure").write_text(
        "some avg10=1.2 avg60=0.5 avg300=0.1 total=987654\n"
        "full avg10=0.7 avg60=0.2 avg300=0.0 total=765432\n"
    )
    sample = c.cgroup_sample(tmp_path)
    assert sample["pressure"]["memory"]["some_total_us"] == 123456
    assert sample["pressure"]["io"]["full_total_us"] == 765432
    assert not c.pressure_reasons({**resources(), "cgroup": sample}, "cuda")


def test_cuda_watchdog_never_spawns_gpu_cli_on_sampling_hot_path(tmp_path, monkeypatch):
    runtime = c.CalibrationRuntime(
        Settings(work_dir=tmp_path),
        RuntimeProfile(),
        "cuda",
        binary=Path("/unused"),
        model_path=Path("/unused"),
        estimate=estimate(),
    )
    watchdog = c.MemoryWatchdog(runtime)
    watchdog.gpu.sample = lambda: {"gpu_free_bytes": 4 * c.GiB, "gpu_utilization_percent": 80}
    monkeypatch.setattr(c, "capture_resources", lambda _: pytest.fail("sampling must use NVML"))
    sample = watchdog.sample()
    assert sample["gpu_utilization_percent"] == 80
    assert sample["elapsed_ms"] >= 0


async def test_cpu_output_projection_report_and_native_probe_consistency(tmp_path, monkeypatch):
    from pd_disaggregation_demo.runtime.engine import LlamaRuntime

    model_path = tmp_path / "model.gguf"
    model_path.touch()
    settings = Settings(model_path=model_path, work_dir=tmp_path)
    expected = (*c.CPU_RECURRENT_PROJECTIONS, r"^output\.weight$=CPU")
    calls = []
    monkeypatch.setattr(c, "ensure_model", lambda *args: model_path)

    def probe(*args, **kwargs):
        calls.append(kwargs["tensor_overrides"])
        return parse_memory_estimate("Host 100 1 1\nMTL0 100 1 1\n")

    monkeypatch.setattr(c, "probe_memory", probe)
    monkeypatch.setattr(
        c,
        "capture_resources",
        lambda _: ResourceSnapshot(
            platform="darwin",
            cpu_count=8,
            ram_total_bytes=64 * c.GiB,
            ram_available_bytes=32 * c.GiB,
            disk_free_bytes=100 * c.GiB,
            metal_recommended_bytes=48 * c.GiB,
        ),
    )
    report = await c.calibrate(
        settings,
        backend="metal",
        output=tmp_path / "report.json",
        ubatches=[4096],
        cpu_recurrent_projections=True,
        cpu_output_projection=True,
        probe_only=True,
        progress=lambda _: None,
    )
    assert report["status"] == "completed"
    assert calls == [expected]
    assert report["placement"]["cpu_output_projection"]
    assert "final logits" in report["placement"]["output_projection_note"]
    native = report["candidates"][0]["native_arguments"]
    assert [
        native[i + 1] for i, argument in enumerate(native) if argument == "--override-tensor"
    ] == list(expected)
    normal = LlamaRuntime(settings, RuntimeProfile(), "metal")._arguments(
        Path("server"), model_path
    )
    assert "--override-tensor" not in normal
