"""Explicitly opted-in, fixed-profile CUDA worker with calibration memory guards."""

import argparse
import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from .backend.app import create_worker_app
from .calibration import (
    CPU_ATTENTION_PROJECTIONS,
    CPU_OUTPUT_PROJECTION,
    CPU_RECURRENT_PROJECTIONS,
    CalibrationRuntime,
    MemoryWatchdog,
    guarded,
    validate_cgroup,
)
from .catalog import ensure_model, get_model
from .config.settings import Settings
from .models import RuntimeProfile
from .runtime.build import ensure_engine
from .runtime.memory_probe import probe_memory


class GuardedWorkerRuntime(CalibrationRuntime):
    """Keep every worker prefill bounded by the watchdog and a finite deadline."""

    def __init__(self, *args, operation_timeout: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_timeout = operation_timeout

    async def configure(self, profile: RuntimeProfile) -> None:
        if profile.model_dump() != self._estimated_profile:
            raise ValueError(
                "Guarded worker profile is fixed to its startup probe; restart and probe again"
            )
        # Never restart after pressure with a failed watchdog or an obsolete admission.
        await guarded(self, self.start(), self.operation_timeout)

    async def prefill(self, token_ids):
        return await guarded(self, super().prefill(token_ids), self.operation_timeout)

    async def health(self) -> dict:
        health = await super().health()
        watcher = self.watchdog
        failure = None
        if watcher and watcher.task and watcher.task.done():
            if watcher.task.cancelled():
                failure = "Calibration watchdog stopped"
            else:
                error = watcher.task.exception()
                failure = str(error) if error else "Calibration watchdog stopped"
            health["status"] = "resource_refused"
        health["calibration"] = {
            "fixed_profile": self._estimated_profile,
            "estimate": asdict(self.estimate),
            "admission": self.calibration_admission,
            "tensor_overrides": list(self.tensor_overrides),
            "cpu_ffn": self.profile.cpu_ffn,
            "operation_timeout_seconds": self.operation_timeout,
            "watchdog": {
                "error": failure,
                "summary": watcher.summary if watcher else {},
                "last": watcher.last if watcher else {},
                "telemetry": watcher.telemetry.describe()
                if watcher and watcher.telemetry
                else None,
            },
        }
        return health


def create_calibration_worker_app(
    settings: Settings,
    profile: RuntimeProfile,
    *,
    cpu_recurrent_projections: bool = False,
    cpu_attention_projections: bool = False,
    cpu_output_projection: bool = False,
    timeout: float = 1200,
    probe_timeout: float = 120,
    telemetry_path: Path | None = None,
):
    """Use existing worker routes with a finite no-swap, reclaimable-mmap lifecycle.

    This is a separate opt-in app; neither ordinary worker admission nor its
    watchdog is weakened. The calibration watchdog reserves 1 GiB GPU memory,
    2 GiB physical host memory and 256 MiB nonreclaimable cgroup headroom.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, probe_timeout)):
        raise ValueError("Worker and probe deadlines must be finite and positive")
    tensor_overrides = (
        (CPU_RECURRENT_PROJECTIONS if cpu_recurrent_projections else ())
        + (CPU_OUTPUT_PROJECTION if cpu_output_projection else ())
        + (CPU_ATTENTION_PROJECTIONS if cpu_attention_projections else ())
    )

    @asynccontextmanager
    async def lifecycle(settings, profile):
        cgroup = await asyncio.to_thread(validate_cgroup)
        spec = get_model(profile.model_id)
        existing = settings.model_path or settings.models_dir / spec.model_id / spec.filename
        if not existing.is_file():
            raise FileNotFoundError(f"Guarded worker requires already-downloaded model: {existing}")
        verify_settings = settings.model_copy(update={"model_path": existing})
        model_path = await asyncio.to_thread(ensure_model, verify_settings, profile.model_id)
        binary = await asyncio.to_thread(ensure_engine, settings, "cuda")
        estimate = await asyncio.to_thread(
            probe_memory,
            settings,
            profile,
            "cuda",
            model_path,
            probe_timeout,
            tensor_overrides=tensor_overrides,
        )
        runtime = GuardedWorkerRuntime(
            settings,
            profile,
            "cuda",
            binary=binary,
            model_path=model_path,
            estimate=estimate,
            cgroup=cgroup,
            tensor_overrides=tensor_overrides,
            operation_timeout=timeout,
        )
        watchdog = MemoryWatchdog(runtime, telemetry_path=telemetry_path)
        try:
            await watchdog.begin()
            await guarded(runtime, runtime.start(), timeout)
            watchdog.mark_phase("worker")
            yield runtime
        finally:
            # Retain guards while the owned native process is being released.
            try:
                await runtime.close()
            finally:
                await watchdog.close()

    return create_worker_app(settings, profile, runtime_lifecycle=lifecycle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--ubatch-size", type=int, required=True)
    parser.add_argument("--cache-type", choices=("f16", "q8_0"), default="q8_0")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--cpu-ffn", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-recurrent-projections", action="store_true")
    parser.add_argument("--cpu-attention-projections", action="store_true")
    parser.add_argument("--cpu-output-projection", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--probe-timeout", type=float, default=120)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = Settings()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if args.model_path:
        settings.model_path = args.model_path
    profile = RuntimeProfile(
        model_id=args.model_id,
        cache_type=args.cache_type,
        context_size=args.context_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        gpu_layers=args.gpu_layers,
        cpu_ffn=args.cpu_ffn,
        threads=args.threads,
    )
    app = create_calibration_worker_app(
        settings,
        profile,
        cpu_recurrent_projections=args.cpu_recurrent_projections,
        cpu_attention_projections=args.cpu_attention_projections,
        cpu_output_projection=args.cpu_output_projection,
        timeout=args.timeout,
        probe_timeout=args.probe_timeout,
        telemetry_path=args.telemetry or settings.work_dir / "guarded-worker.telemetry.jsonl",
    )
    import uvicorn

    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.worker_port)
