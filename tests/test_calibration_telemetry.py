import json
from types import SimpleNamespace

import pytest

from pd_disaggregation_demo.calibration_telemetry import NvmlSession, TelemetryRecorder


def test_time_weighted_utilization_and_physical_reads_are_not_sample_averages(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "samples.jsonl")
    for elapsed, util, read in [(0, 0, 100), (100, 100, 200), (1000, 0, 1100)]:
        recorder.record(
            {"elapsed_ms": elapsed, "gpu_utilization_percent": util, "native_read_bytes": read},
            "measured:0",
        )
    result = recorder.describe()
    recorder.close()
    summary = result["phases"]["measured:0"]
    assert summary["time_weighted_gpu_utilization_percent"] == 90
    assert summary["fraction_gpu_utilization_below_20_percent"] == 0.1
    assert summary["physical_read_bytes_delta"] == 1000
    assert summary["physical_read_bytes_per_second_max"] == 1000
    assert summary["actual_sample_period_mean_ms"] == 500
    assert summary["gap_max_ms"] == 900
    lines = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert lines[0]["columns"] == list(TelemetryRecorder.COLUMNS)
    assert len(lines) == 4
    assert lines[-1][0] == 1000


def test_phase_boundaries_do_not_blend_load_and_inference(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "samples.jsonl")
    recorder.mark("load", 0)
    recorder.record({"elapsed_ms": 0, "gpu_utilization_percent": 0}, "load")
    recorder.record({"elapsed_ms": 100, "gpu_utilization_percent": 0}, "load")
    recorder.mark("measured:0", 150)
    recorder.record({"elapsed_ms": 150, "gpu_utilization_percent": 100}, "measured:0")
    recorder.record({"elapsed_ms": 250, "gpu_utilization_percent": 100}, "measured:0")
    result = recorder.describe()
    recorder.close()
    assert result["phases"]["load"]["time_weighted_gpu_utilization_percent"] == 0
    assert result["phases"]["measured:0"]["time_weighted_gpu_utilization_percent"] == 100
    assert result["phase_markers"][1] == {"phase": "measured:0", "elapsed_ms": 150}


def test_series_is_bounded_and_missing_gpu_metrics_stay_unknown(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "samples.jsonl", max_samples=2)
    for elapsed in range(10):
        recorder.record({"elapsed_ms": elapsed * 100, "native_read_bytes": elapsed * 10}, "load")
    result = recorder.describe()
    recorder.close()
    assert result["recorded_samples"] == 2 and result["dropped_samples"] == 8
    assert result["truncated"]
    assert result["summary"]["sample_count"] == 10
    assert result["summary"]["time_weighted_gpu_utilization_percent"] is None
    assert len(recorder.path.read_text().splitlines()) == 3


def test_persistent_nvml_optional_fields_are_not_mandatory_for_safety(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda _: "gpu0",
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(total=8000, free=2000, used=6000),
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=80, memory=40),
        nvmlDeviceGetClockInfo=lambda _, clock: 1500,
        nvmlDeviceGetPowerUsage=lambda _: 75000,
        NVML_CLOCK_SM=0,
        NVML_CLOCK_MEM=1,
    )
    monkeypatch.setattr(
        "pd_disaggregation_demo.calibration_telemetry.importlib.import_module", lambda _: fake
    )
    session = NvmlSession()
    session.open()
    first = session.sample()
    session.sample()
    session.close()
    assert calls == ["init", "shutdown"]
    assert first["gpu_free_bytes"] == 2000
    assert first["gpu_utilization_percent"] == 80
    assert first["gpu_power_watts"] == 75
    assert first["gpu_pcie_rx_kb_per_second"] is None
    assert "gpu_pcie_rx_kb_per_second" in first["gpu_telemetry_unavailable"]


def test_missing_mandatory_nvml_memory_measurement_fails_closed():
    session = NvmlSession()
    session.module = SimpleNamespace(
        nvmlDeviceGetMemoryInfo=lambda _: (_ for _ in ()).throw(
            RuntimeError("mandatory GPU memory unavailable")
        )
    )
    session.handle = "gpu0"
    with pytest.raises(RuntimeError, match="mandatory"):
        session.sample()
