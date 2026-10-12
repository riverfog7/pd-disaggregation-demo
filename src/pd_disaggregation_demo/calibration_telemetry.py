"""Bounded GPU/disk telemetry for an explicitly owned calibration runtime."""

import contextlib
import importlib
import json
import os
from pathlib import Path


class NvmlSession:
    """Keep one NVML session; never launch nvidia-smi in the sampling hot path."""

    def __init__(self):
        self.module = None
        self.handle = None

    def open(self) -> None:
        self.module = importlib.import_module("pynvml")
        self.module.nvmlInit()
        try:
            self.handle = self.module.nvmlDeviceGetHandleByIndex(0)
        except BaseException:
            self.close()
            raise

    def sample(self) -> dict:
        if self.module is None or self.handle is None:
            raise RuntimeError("CUDA watchdog requires its persistent NVML session")
        nvml, handle = self.module, self.handle
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)  # Mandatory safety measurement.
        data = {
            "gpu_total_bytes": memory.total,
            "gpu_free_bytes": memory.free,
            "gpu_used_bytes": memory.used,
        }
        optional = {
            "gpu_utilization_percent": lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "gpu_memory_utilization_percent": lambda: (
                nvml.nvmlDeviceGetUtilizationRates(handle).memory
            ),
            "gpu_sm_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
            "gpu_memory_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(
                handle, nvml.NVML_CLOCK_MEM
            ),
            "gpu_power_watts": lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "gpu_pcie_rx_kb_per_second": lambda: nvml.nvmlDeviceGetPcieThroughput(
                handle, nvml.NVML_PCIE_UTIL_RX_BYTES
            ),
        }
        unavailable = {}
        for key, getter in optional.items():
            try:
                data[key] = getter()
            except Exception as error:
                data[key] = None
                unavailable[key] = f"{type(error).__name__}: {error}"
        if unavailable:
            data["gpu_telemetry_unavailable"] = unavailable
        return data

    def close(self) -> None:
        if self.module is not None:
            with contextlib.suppress(Exception):
                self.module.nvmlShutdown()
        self.module = self.handle = None


class TelemetryRecorder:
    """Write compact JSONL rows and keep only aggregate statistics in memory."""

    COLUMNS = (
        "elapsed_ms",
        "phase",
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_sm_clock_mhz",
        "gpu_memory_clock_mhz",
        "gpu_power_watts",
        "gpu_pcie_rx_kb_per_second",
        "native_read_bytes",
        "native_major_faults",
        "native_process_state",
        "native_rss_bytes",
        "gpu_free_bytes",
        "ram_available_bytes",
        "cgroup_current_bytes",
        "cgroup_anon_bytes",
        "cgroup_file_bytes",
        "cgroup_memory_some_total_us",
        "cgroup_memory_full_total_us",
        "cgroup_io_some_total_us",
        "cgroup_io_full_total_us",
    )

    def __init__(self, path: Path, *, max_samples: int = 50_000, max_bytes: int = 16 * 1024**2):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", buffering=1)
        self.file.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "columns": self.COLUMNS,
                    "clock": "monotonic milliseconds relative to watchdog creation",
                }
            )
            + "\n"
        )
        self.max_samples, self.max_bytes = max_samples, max_bytes
        self.recorded, self.dropped = 0, 0
        self.previous = None
        self.stats = {}
        self.phases = {}
        self.markers = []
        self.unavailable = {}

    def mark(self, phase: str, elapsed_ms: float) -> None:
        self.markers.append({"phase": phase, "elapsed_ms": elapsed_ms})

    @staticmethod
    def _accumulate(stats: dict, previous: dict | None, sample: dict) -> None:
        stats["sample_count"] = stats.get("sample_count", 0) + 1
        if previous is None:
            return
        dt = sample["elapsed_ms"] - previous["elapsed_ms"]
        if dt <= 0:
            return
        stats["gap_total_ms"] = stats.get("gap_total_ms", 0) + dt
        stats["gap_count"] = stats.get("gap_count", 0) + 1
        stats["gap_max_ms"] = max(dt, stats.get("gap_max_ms", 0))
        stats["gap_min_ms"] = min(dt, stats.get("gap_min_ms", dt))
        util = previous.get("gpu_utilization_percent")
        if util is not None:
            stats["gpu_coverage_ms"] = stats.get("gpu_coverage_ms", 0) + dt
            stats["gpu_percent_ms"] = stats.get("gpu_percent_ms", 0) + util * dt
            stats["gpu_below_20_ms"] = stats.get("gpu_below_20_ms", 0) + (dt if util < 20 else 0)
        left, right = previous.get("native_read_bytes"), sample.get("native_read_bytes")
        if left is not None and right is not None and right >= left:
            delta = right - left
            stats["physical_read_bytes_delta"] = stats.get("physical_read_bytes_delta", 0) + delta
            stats["physical_read_bytes_per_second_max"] = max(
                delta * 1000 / dt, stats.get("physical_read_bytes_per_second_max", 0)
            )

    def record(self, sample: dict, phase: str) -> None:
        row = {key: sample.get(key) for key in self.COLUMNS}
        row["phase"] = phase
        group = sample.get("cgroup", {})
        for name in ("current_bytes", "anon_bytes", "file_bytes"):
            row["cgroup_" + name] = group.get(name)
        for kind, values in group.get("pressure", {}).items():
            for key, value in values.items():
                row[f"cgroup_{kind}_{key}"] = value
        self._accumulate(self.stats, self.previous, row)
        phase_stats = self.phases.setdefault(phase, {})
        if self.previous is not None and self.previous["phase"] != phase:
            self._accumulate(phase_stats, None, row)
        else:
            self._accumulate(phase_stats, self.previous, row)
        self.previous = row
        self.unavailable.update(sample.get("gpu_telemetry_unavailable", {}))
        if sample.get("native_metrics_unavailable"):
            self.unavailable["native_process_metrics"] = sample["native_metrics_unavailable"]
        line = (
            json.dumps([row[key] for key in self.COLUMNS], separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        if (
            self.recorded < self.max_samples
            and self.file.tell() + len(line.encode()) <= self.max_bytes
        ):
            self.file.write(line)
            self.recorded += 1
        else:
            self.dropped += 1

    @staticmethod
    def summarized(stats: dict) -> dict:
        result = {
            key: value
            for key, value in stats.items()
            if key not in ("gpu_percent_ms", "gpu_below_20_ms", "gap_total_ms")
        }
        coverage = stats.get("gpu_coverage_ms", 0)
        result["time_weighted_gpu_utilization_percent"] = (
            stats.get("gpu_percent_ms", 0) / coverage if coverage else None
        )
        result["fraction_gpu_utilization_below_20_percent"] = (
            stats.get("gpu_below_20_ms", 0) / coverage if coverage else None
        )
        gaps = stats.get("gap_count", 0)
        result["actual_sample_period_mean_ms"] = (
            stats.get("gap_total_ms", 0) / gaps if gaps else None
        )
        return result

    def describe(self) -> dict:
        return {
            "jsonl_path": str(self.path),
            "columns": self.COLUMNS,
            "recorded_samples": self.recorded,
            "dropped_samples": self.dropped,
            "max_samples": self.max_samples,
            "max_bytes": self.max_bytes,
            "truncated": self.dropped > 0,
            "phase_markers": list(self.markers),
            "summary": self.summarized(self.stats),
            "phases": {phase: self.summarized(stats) for phase, stats in self.phases.items()},
            "unavailable": dict(self.unavailable),
            "interpretation": "Process read_bytes counts physical reads, including readahead. "
            "Utilization and IO/pressure correlation cannot establish full causality "
            "without a trace.",
        }

    def close(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
