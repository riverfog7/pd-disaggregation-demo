"""Opt-in, reserve-guarded native microbatch calibration.

This deliberately does not change ordinary worker admission. CUDA trials permit
reclaimable mmap weights only inside an inherited, finite, no-swap cgroup v2.
The watchdog can stop only this harness's owned llama-server process.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Coroutine
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from .artifacts import delete_artifact, ensure_state_disk_space
from .benchmarks.runner import _check_result, exact_tokens
from .calibration_telemetry import NvmlSession, TelemetryRecorder
from .catalog import ensure_model, get_model
from .config.settings import Settings
from .models import GenerationRequest, RuntimeFingerprint, RuntimeProfile
from .resources import AdmissionResult, capture_resources
from .runtime.build import ensure_engine
from .runtime.engine import LlamaRuntime, elapsed_ms
from .runtime.memory_probe import MemoryEstimate, probe_memory, tensor_override_arguments

GiB = 1024**3
MiB = 1024**2
DEFAULT_UBATCHES = (128, 256, 512, 1024, 2048, 4096, 8192)
STAGING_BYTES = 512 * MiB
CPU_RECURRENT_PROJECTIONS = (r"^blk\.[0-9]+\.(attn_qkv|attn_gate|ssm_out)\.weight$=CPU",)
CPU_OUTPUT_PROJECTION = (r"^output\.weight$=CPU",)


class PressureError(RuntimeError):
    """A live measurement violated a mandatory calibration reserve."""


def current_cgroup() -> Path:
    """Resolve the caller's real cgroup through the cgroup2 mount namespace."""
    membership = next(
        (
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if membership is None:
        raise ValueError("CUDA calibration requires cgroup v2")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] != "cgroup2":
            continue
        fields = before.split()
        root, mount = fields[3], fields[4]
        # Linux mountinfo escapes whitespace and backslashes with octal sequences.
        root = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), root)
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), mount)
        if membership == "/":
            candidate = Path(mount)
        elif membership == root or membership.startswith(root.rstrip("/") + "/"):
            candidate = Path(mount) / membership[len(root) :].lstrip("/")
        else:
            continue
        if (candidate / "memory.current").is_file():
            return candidate.resolve()
    raise ValueError("Cannot resolve the invoking process cgroup v2")


def _key_values(path: Path) -> dict[str, int]:
    return {
        key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())
    }


def cgroup_sample(root: Path) -> dict[str, Any]:
    """Count clean file pages as reclaimable, never anonymous or dirty pages."""
    limits, swaps = [], []
    node = root
    while (node / "memory.current").exists():
        maximum = (node / "memory.max").read_text().strip()
        swap = (node / "memory.swap.max").read_text().strip()
        if maximum != "max":
            limits.append(int(maximum))
        if swap != "max":
            swaps.append(int(swap))
        if node.parent == node:
            break
        node = node.parent
    stat = _key_values(root / "memory.stat")
    current = int((root / "memory.current").read_text())
    file_bytes = stat.get("file", 0)
    reclaimable = max(
        0,
        file_bytes
        - stat.get("shmem", 0)
        - stat.get("file_dirty", 0)
        - stat.get("file_writeback", 0),
    )
    limit = min(limits) if limits else None
    # Ancestor totals can contain sibling usage; include their unreclaimable
    # headroom so a leaf never treats the full shared parent cap as available.
    headrooms = []
    node = root
    while (node / "memory.current").exists():
        maximum = (node / "memory.max").read_text().strip()
        if maximum != "max":
            ancestor_stat = _key_values(node / "memory.stat")
            clean = max(
                0,
                ancestor_stat.get("file", 0)
                - ancestor_stat.get("shmem", 0)
                - ancestor_stat.get("file_dirty", 0)
                - ancestor_stat.get("file_writeback", 0),
            )
            used = int((node / "memory.current").read_text()) - clean
            headrooms.append(int(maximum) - max(0, used))
        if node.parent == node:
            break
        node = node.parent
    pressure = {}
    for kind in ("memory", "io"):
        try:
            pressure[kind] = {
                fields[0] + "_total_us": int(
                    next(value[6:] for value in fields[1:] if value.startswith("total="))
                )
                for fields in (
                    line.split() for line in (root / f"{kind}.pressure").read_text().splitlines()
                )
            }
        except (OSError, ValueError, StopIteration):
            pass
    return {
        "path": str(root),
        "limit_bytes": limit,
        "swap_limit_bytes": min(swaps) if swaps else None,
        "current_bytes": current,
        "anon_bytes": stat.get("anon", 0),
        "file_bytes": file_bytes,
        "reclaimable_bytes": reclaimable,
        "nonreclaimable_bytes": max(0, current - reclaimable),
        "nonreclaimable_headroom_bytes": min(headrooms) if headrooms else None,
        "events": _key_values(root / "memory.events"),
        "pressure": pressure,
    }


def validate_cgroup(root: Path | None = None) -> Path:
    actual = current_cgroup()
    if root is not None and root.resolve() != actual:
        raise ValueError(
            "--cgroup must be the invoking process cgroup, inherited by the native child"
        )
    sample = cgroup_sample(actual)
    if sample["limit_bytes"] is None or sample["limit_bytes"] <= 0:
        raise ValueError("CUDA calibration requires a finite cgroup memory.max")
    if sample["swap_limit_bytes"] != 0:
        raise ValueError("CUDA calibration requires effective memory.swap.max=0")
    return actual


def preflight(
    estimate: MemoryEstimate,
    resources: dict,
    backend: str,
    arguments: list[str],
    cgroup: dict | None,
) -> dict:
    device = sum(component.total_bytes for component in estimate.devices.values())
    host = estimate.host
    # State export is separately checked against disk capacity by the runtime.
    required_host = host.context_bytes + host.compute_bytes + STAGING_BYTES
    reasons = []
    budgets = {
        "native_host_total_bytes": host.total_bytes,
        "native_device_total_bytes": device,
        "nonreclaimable_host_required_bytes": required_host,
        "staging_bytes": STAGING_BYTES,
    }
    if not estimate.devices:
        reasons.append("Native estimate has no accelerator device")
    if backend == "cuda":
        mmap = (
            "--no-host" in arguments
            and "--no-repack" in arguments
            and "--load-mode" in arguments
            and arguments[arguments.index("--load-mode") + 1] == "mmap"
        )
        if not mmap:
            reasons.append(
                "Reclaimable CUDA weights require --no-host --no-repack --load-mode mmap"
            )
        gpu_free = resources.get("gpu_free_bytes")
        budgets["gpu_required_with_reserve_bytes"] = device + GiB + 256 * MiB
        if gpu_free is None or device + GiB + 256 * MiB > gpu_free:
            reasons.append(
                "Native device estimate plus 1 GiB reserve and 256 MiB driver cushion "
                "exceeds measured GPU free memory"
            )
        if required_host + 2 * GiB > resources["ram_available_bytes"]:
            reasons.append("Nonreclaimable host estimate plus 2 GiB exceeds physical availability")
        if not cgroup or cgroup.get("limit_bytes") is None or cgroup.get("swap_limit_bytes") != 0:
            reasons.append("CUDA requires a finite no-swap cgroup v2")
        elif required_host + GiB > cgroup["nonreclaimable_headroom_bytes"]:
            reasons.append("Nonreclaimable host estimate plus 1 GiB exceeds cgroup headroom")
    elif backend == "metal":
        total = host.total_bytes + device + STAGING_BYTES
        budgets["unified_required_with_reserve_bytes"] = total + 6 * GiB
        if total + 6 * GiB > resources["ram_available_bytes"]:
            reasons.append(
                "Sum of native Host and Metal estimates plus staging and 6 GiB reserve "
                "exceeds unified memory availability"
            )
        working = resources.get("metal_recommended_bytes")
        if working is None or device + GiB > working:
            reasons.append("Native Metal estimate plus 1 GiB exceeds measured working-set limit")
    else:
        reasons.append("Calibration supports Metal and CUDA only")
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "budgets": budgets,
        "resources": resources,
        "cgroup": cgroup,
    }


def pressure_reasons(sample: dict, backend: str) -> list[str]:
    reasons = []
    reserve = 6 * GiB if backend == "metal" else 2 * GiB
    if sample["ram_available_bytes"] < reserve:
        reasons.append(f"Physical host available memory fell below {reserve // GiB} GiB")
    if backend == "cuda":
        if sample.get("gpu_free_bytes") is None or sample["gpu_free_bytes"] < GiB:
            reasons.append("GPU free memory fell below 1 GiB or became unmeasurable")
        cgroup = sample.get("cgroup")
        if not cgroup or cgroup.get("nonreclaimable_headroom_bytes") is None:
            reasons.append("Cgroup headroom became unmeasurable")
        elif cgroup["nonreclaimable_headroom_bytes"] < 256 * MiB:
            reasons.append("Nonreclaimable cgroup headroom fell below 256 MiB")
        if cgroup and cgroup.get("swap_limit_bytes") != 0:
            reasons.append("Cgroup swap prohibition changed")
    return reasons


def effective_from_log(log: str) -> dict:
    """Leave unobserved flags unknown; keep the actual diagnostic source."""
    result = {
        "observed_from": "native log",
        "effective_n_ctx": None,
        "effective_n_batch": None,
        "effective_n_ubatch": None,
        "flash_attention": None,
    }
    for field, name in [
        ("n_ctx", "effective_n_ctx"),
        ("n_batch", "effective_n_batch"),
        ("n_ubatch", "effective_n_ubatch"),
    ]:
        values = re.findall(rf"\b{field}\s*=\s*(\d+)\b", log)
        if values:
            result[name] = int(values[-1])
    flags = re.findall(r"\bflash_attn\s*=\s*(\S+)", log)
    if flags:
        value = flags[-1].lower().strip(",")
        if value in ("1", "on", "true", "enabled"):
            result["flash_attention"] = True
        elif value in ("0", "off", "false", "disabled"):
            result["flash_attention"] = False
    return result


def validate_effective(observed: dict, profile: RuntimeProfile) -> None:
    """Configured flags cannot stand in for an observed native configuration."""
    for key, required in (
        ("effective_n_batch", profile.batch_size),
        ("effective_n_ubatch", profile.ubatch_size),
    ):
        if observed.get(key) != required:
            raise RuntimeError(f"Native {key}={observed.get(key)!r}; requested {required}")
    context = observed.get("effective_n_ctx")
    if not isinstance(context, int) or context < profile.context_size:
        raise RuntimeError("Native effective context is missing or smaller than requested")
    if observed.get("flash_attention") is not True:
        raise RuntimeError("Native FlashAttention was not observed enabled")


def _log_tail(path: Path, maximum: int = 2 * MiB) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - maximum))
        return handle.read(maximum).decode("utf-8", errors="replace")


class CalibrationRuntime(LlamaRuntime):
    """Explicit calibration-only admission; normal request/state checks are inherited.

    The caller must run a MemoryWatchdog for the entire load and operation lifetime.
    The sweep does this automatically; later experiments can use guarded().
    """

    def __init__(
        self,
        settings,
        profile,
        backend,
        *,
        binary: Path,
        model_path: Path,
        estimate: MemoryEstimate,
        cgroup: Path | None = None,
        tensor_overrides: tuple[str, ...] = (),
    ):
        super().__init__(settings, profile, backend)
        self.binary, self.model_path, self.estimate = binary, model_path, estimate
        self._estimated_profile = self.profile.model_dump()
        self.tensor_overrides = tensor_overrides
        self.cgroup = cgroup
        self.calibration_admission: dict | None = None
        self.watchdog: MemoryWatchdog | None = None

    @property
    def log_path(self) -> Path:
        return self.settings.work_dir / f"llama-{self.backend}.log"

    def _arguments(self, binary: Path, model_path: Path) -> list[str]:
        return [
            *super()._arguments(binary, model_path),
            *tensor_override_arguments(self.tensor_overrides),
            "--verbosity",
            "4",
        ]

    async def start(self) -> None:
        if self.profile.model_dump() != self._estimated_profile:
            raise RuntimeError("Calibration estimate is bound to its original profile; probe again")
        if self._client and self._process_reusable():
            return
        if not self.watchdog or not self.watchdog.task or self.watchdog.task.done():
            raise RuntimeError("CalibrationRuntime.start requires an active MemoryWatchdog")
        if self._client or self._process:
            await self.close()
        started = time.perf_counter()
        if self.backend == "cuda":
            self.cgroup = validate_cgroup(self.cgroup)
        snapshot = await asyncio.to_thread(capture_resources, self.settings.work_dir)
        group = await asyncio.to_thread(cgroup_sample, self.cgroup) if self.cgroup else None
        arguments = self._arguments(self.binary, self.model_path)
        self.calibration_admission = preflight(
            self.estimate, snapshot.model_dump(mode="json"), self.backend, arguments, group
        )
        if not self.calibration_admission["allowed"]:
            raise PressureError("; ".join(self.calibration_admission["reasons"]))
        spec = get_model(self.profile.model_id)
        cache_element = 34 / 32 if self.profile.cache_type == "q8_0" else 2
        state_bytes = (
            math.ceil(
                2
                * spec.attention_layers
                * spec.kv_heads
                * spec.head_dim
                * self.profile.context_size
                * cache_element
            )
            + spec.recurrent_bytes
        )
        await asyncio.to_thread(
            ensure_state_disk_space,
            self.settings.work_dir,
            state_bytes,
            self.settings.disk_reserve_bytes,
        )
        self._admission = AdmissionResult(
            allowed=True, budgets={"state_bytes": state_bytes}, resources=snapshot
        )
        self.fingerprint = RuntimeFingerprint(
            engine_revision=self.settings.engine_revision,
            model_sha256=spec.sha256,
            cache_type_k=self.profile.cache_type,
            cache_type_v=self.profile.cache_type,
            context_size=self.profile.context_size,
        )
        self._slots_dir.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", self.settings.engine_port))
            self._process = await asyncio.create_subprocess_exec(
                *arguments, stdout=self._log, stderr=self._log
            )
            if self.cgroup:
                # The child inherits membership; don't silently accept a fake or unrelated group.
                own = Path("/proc/self/cgroup").read_text()
                child = Path(f"/proc/{self._process.pid}/cgroup").read_text()
                if child != own:
                    raise RuntimeError("Native child did not inherit the guarded cgroup")
            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self.settings.engine_port}",
                timeout=self.settings.request_timeout,
                trust_env=False,
            )
            deadline = time.monotonic() + self.settings.startup_timeout
            while time.monotonic() < deadline:
                if self._process.returncode is not None:
                    raise RuntimeError(f"Owned llama-server exited ({self._process.returncode})")
                try:
                    response = await self._client.get("/health", timeout=1)
                    if response.status_code == 200:
                        await self._capture_native_props()
                        observed = effective_from_log(_log_tail(self.log_path))
                        self._effective_settings = {
                            **(self._effective_settings or {}),
                            "native_log": observed,
                        }
                        # Context rounding is allowed; batch and microbatch must match.
                        validate_effective(observed, self.profile)
                        self.startup_ms = elapsed_ms(started)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
            raise TimeoutError("Calibration native startup timed out")
        except BaseException as error:
            await self._close_after_failure(error)
            raise


async def stop_owned(runtime: LlamaRuntime) -> None:
    process = runtime._process
    if process is not None and process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), 2)


class MemoryWatchdog:
    def __init__(self, runtime: CalibrationRuntime, *, telemetry_path: Path | None = None):
        self.runtime = runtime
        runtime.watchdog = self
        self.task: asyncio.Task | None = None
        self.summary: dict[str, Any] = {"sample_count": 0, "requested_period_ms": 100}
        self.last: dict = {}
        self.events_before: dict = {}
        self.phase_summary: dict = {"sample_count": 0}
        self.started = time.monotonic()
        self.phase = "load"
        self.gpu = NvmlSession() if runtime.backend == "cuda" else None
        self.telemetry = TelemetryRecorder(telemetry_path) if telemetry_path else None
        if self.telemetry:
            self.telemetry.mark(self.phase, 0)

    def mark_phase(self, phase: str) -> None:
        self.phase = phase
        if self.telemetry:
            self.telemetry.mark(phase, (time.monotonic() - self.started) * 1000)

    def sample(self) -> dict:
        if self.gpu:
            memory = psutil.virtual_memory()
            data = {
                "ram_available_bytes": memory.available,
                "ram_total_bytes": memory.total,
                **self.gpu.sample(),
            }
        else:
            data = capture_resources(self.runtime.settings.work_dir).model_dump(mode="json")
        data["physical_ram_available_bytes"] = psutil.virtual_memory().available
        data["ram_available_bytes"] = data["physical_ram_available_bytes"]
        if self.runtime.cgroup:
            data["cgroup"] = cgroup_sample(self.runtime.cgroup)
        process = self.runtime._process
        if process and process.returncode is None:
            try:
                native = psutil.Process(process.pid)
                data["native_rss_bytes"] = native.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as error:
                data.setdefault("native_metrics_unavailable", f"{type(error).__name__}: {error}")
            if self.runtime.backend == "cuda":
                try:
                    native = psutil.Process(process.pid)
                    data["native_read_bytes"] = native.io_counters().read_bytes
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as error:
                    data.setdefault(
                        "native_metrics_unavailable", f"{type(error).__name__}: {error}"
                    )
                try:
                    stat = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                    data["native_major_faults"] = int(stat[9])
                    data["native_process_state"] = stat[0]
                except (OSError, ValueError, IndexError) as error:
                    data.setdefault(
                        "native_metrics_unavailable", f"{type(error).__name__}: {error}"
                    )
        data["elapsed_ms"] = (time.monotonic() - self.started) * 1000
        return data

    def retain(self, sample: dict) -> None:
        self.last = sample
        self._retain_into(self.summary, sample)
        self._retain_into(self.phase_summary, sample)
        if self.telemetry:
            self.telemetry.record(sample, self.phase)

    def _retain_into(self, summary: dict, sample: dict) -> None:
        summary["sample_count"] += 1
        for name in ("ram_available_bytes", "gpu_free_bytes"):
            value = sample.get(name)
            if value is not None:
                key = name.replace("_bytes", "_min_bytes")
                summary[key] = min(value, summary.get(key, value))
        for name in ("native_rss_bytes", "native_read_bytes", "native_major_faults"):
            value = sample.get(name)
            if value is not None:
                key = name + "_max"
                summary[key] = max(value, summary.get(key, value))
        group = sample.get("cgroup", {})
        for name in (
            "current_bytes",
            "anon_bytes",
            "file_bytes",
            "reclaimable_bytes",
            "nonreclaimable_bytes",
        ):
            if name in group:
                key = "cgroup_" + name + "_max"
                summary[key] = max(group[name], summary.get(key, group[name]))
        if group.get("nonreclaimable_headroom_bytes") is not None:
            value = group["nonreclaimable_headroom_bytes"]
            key = "cgroup_nonreclaimable_headroom_min_bytes"
            summary[key] = min(value, summary.get(key, value))

    def check_sample(self, sample: dict) -> None:
        self.retain(sample)
        reasons = pressure_reasons(sample, self.runtime.backend)
        events = sample.get("cgroup", {}).get("events", {})
        if any(
            events.get(key, 0) > self.events_before.get(key, 0)
            for key in ("oom", "oom_kill", "oom_group_kill")
        ):
            reasons.append("Cgroup OOM event counter increased")
        if self.runtime.log_path.exists() and self.runtime.log_path.stat().st_size > 32 * MiB:
            reasons.append("Native diagnostic log exceeded its 32 MiB budget")
        if reasons:
            self.summary["pressure"] = reasons
            raise PressureError("; ".join(reasons))

    async def run(self) -> None:
        try:
            while True:
                started = time.monotonic()
                sample = await asyncio.to_thread(self.sample)
                self.check_sample(sample)
                duration = (time.monotonic() - started) * 1000
                self.summary["sample_duration_max_ms"] = max(
                    duration, self.summary.get("sample_duration_max_ms", 0)
                )
                await asyncio.sleep(max(0, 0.1 - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception:
            await stop_owned(self.runtime)
            raise

    async def begin(self) -> None:
        # A synchronous initial sample prevents loading before the first guard result.
        if self.gpu:
            await asyncio.to_thread(self.gpu.open)
        sample = await asyncio.to_thread(self.sample)
        self.events_before = sample.get("cgroup", {}).get("events", {})
        self.check_sample(sample)
        self.task = asyncio.create_task(self.run())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        if self.gpu:
            await asyncio.to_thread(self.gpu.close)
        if self.telemetry:
            self.telemetry.close()


async def guarded(runtime: CalibrationRuntime, operation: Coroutine, timeout: float):
    """Race a native operation against the continuous watchdog and a hard deadline."""
    if not runtime.watchdog or not runtime.watchdog.task:
        operation.close()
        raise RuntimeError("Native operation requires an active watchdog")
    task = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait(
            (task, runtime.watchdog.task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if runtime.watchdog.task in done:
            await runtime.watchdog.task
        if task in done:
            return await task
        raise TimeoutError(f"Calibration operation exceeded {timeout:g} seconds")
    except BaseException:
        await stop_owned(runtime)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, 15)
        raise


def atomic_report(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


async def measure(runtime: CalibrationRuntime, tokens: list[int], output_tokens: int) -> dict:
    if runtime.backend == "cuda":
        artifact = await runtime.prefill(tokens)
        try:
            manifest = artifact.manifest.model_dump(mode="json")
            if manifest["prefix_tokens"] != len(tokens) - 1:
                raise RuntimeError("Calibration state boundary mismatch")
            for key in ("native_prefill_ms", "prefill_tps"):
                value = manifest["timings"].get(key)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Native prefill timing {key} must be finite and positive")
            # The candidate records the canonical prompt hash. Avoid copying the
            # same full token list into every trial's report.
            manifest.pop("token_ids")
            return {
                "operation": "prefill_export",
                "manifest": manifest,
                "native_prompt_n": len(tokens) - 1,
                "native_cache_n": 0,
                "native_saved_n": manifest["prefix_tokens"],
                "actual_state_bytes": artifact.state_path.stat().st_size,
                "discarded_predict_tokens": 1,
                "canonical_prompt_tokens": len(tokens),
                "largest_exercised_prompt_chunk": min(runtime.profile.ubatch_size, len(tokens) - 1),
            }
        finally:
            delete_artifact(artifact)
    result = None
    async with contextlib.aclosing(
        runtime.generate(
            GenerationRequest(
                token_ids=tokens, max_tokens=output_tokens, ignore_eos=True, temperature=0, seed=0
            )
        )
    ) as stream:
        async for event in stream:
            if event.kind == "error":
                raise RuntimeError(event.text or str(event.data))
            if event.kind == "result":
                result = event.result
    if result is None:
        raise RuntimeError("Calibration generation returned no terminal result")
    _check_result(result, len(tokens), output_tokens, "local", runtime.profile.model_id)
    return {"operation": "local_generate", "result": result.model_dump(mode="json")}


async def calibrate(
    settings: Settings,
    *,
    backend: str,
    output: Path,
    ubatches=DEFAULT_UBATCHES,
    repeats: int = 3,
    prompt_tokens: int = 8192,
    output_tokens: int = 128,
    cpu_ffn: int | None = None,
    model_id: str = "qwen3.5-27b",
    timeout: float = 1200,
    probe_only: bool = False,
    cgroup: Path | None = None,
    cpu_recurrent_projections: bool = False,
    cpu_output_projection: bool = False,
    progress: Callable[[str], None] = print,
) -> dict:
    if backend not in ("cuda", "metal"):
        raise ValueError("Backend must be cuda or metal")
    if repeats < 1 or repeats > 100 or not 2 <= prompt_tokens <= 262016:
        raise ValueError("Repeats must be 1..100 and prompt tokens 2..262016")
    if not 1 <= output_tokens <= 128 or timeout <= 0:
        raise ValueError("Output tokens must be 1..128 and timeout must be positive")
    ubatches = tuple(sorted(set(ubatches)))
    if not ubatches or any(not 1 <= value <= prompt_tokens for value in ubatches):
        raise ValueError("Microbatches must be positive and no larger than logical batch")
    if len(ubatches) > 32:
        raise ValueError("At most 32 microbatch candidates per calibration report")
    profile = RuntimeProfile(
        model_id=model_id,
        cache_type="q8_0",
        context_size=max(256, prompt_tokens + output_tokens),
        batch_size=prompt_tokens,
        ubatch_size=ubatches[0],
        cpu_ffn=(64 if backend == "cuda" else 0) if cpu_ffn is None else cpu_ffn,
    )
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    tensor_overrides = (CPU_RECURRENT_PROJECTIONS if cpu_recurrent_projections else ()) + (
        CPU_OUTPUT_PROJECTION if cpu_output_projection else ()
    )
    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "probe_only": probe_only,
        "requested_profile": profile.model_dump(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "repeats": repeats,
        "trial_timeout_seconds": timeout,
        "placement": {
            "cpu_recurrent_projections": cpu_recurrent_projections,
            "cpu_output_projection": cpu_output_projection,
            "tensor_overrides": list(tensor_overrides),
            "cpu_ffn": profile.cpu_ffn,
            "output_projection_note": "CPU output-weight placement may compute final logits on CPU"
            if cpu_output_projection
            else None,
        },
        "candidates": [],
        "notes": [
            "Estimated safe is not measured safe. Watchdog sampling cannot guarantee no OOM.",
            "CUDA mmap file pages may reclaim and refault; read bytes and major faults recorded.",
            "Ordinary worker admission is unchanged. Calibration terminates only its native child.",
            "Warmup uses the full canonical prompt; every request starts with an empty cache.",
        ],
    }
    atomic_report(output, report)
    try:
        if backend == "cuda":
            cgroup = validate_cgroup(cgroup)
        model = get_model(model_id)
        existing = settings.model_path or settings.models_dir / model.model_id / model.filename
        if not existing.is_file():
            raise FileNotFoundError(f"Calibration requires already-downloaded model: {existing}")
        # Force the existing-file path: this harness must never download weights.
        verify_settings = settings.model_copy(update={"model_path": existing})
        model_path = await asyncio.to_thread(ensure_model, verify_settings, model_id)
        binary = None if probe_only else await asyncio.to_thread(ensure_engine, settings, backend)
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = settings.git_revision
        report["environment"] = {
            "engine_revision": settings.engine_revision,
            "git_revision": revision,
            "platform": platform.platform(),
            "model_path": str(model_path),
            "model_sha256": model.sha256,
            "native_binary": str(binary) if binary else None,
            "image_id": os.environ.get("PDD_CALIBRATION_IMAGE_ID"),
            "source_revision": os.environ.get("PDD_CALIBRATION_SOURCE_REVISION", revision),
            "calibration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "telemetry_source_sha256": hashlib.sha256(
                Path(__file__).with_name("calibration_telemetry.py").read_bytes()
            ).hexdigest(),
            "cgroup_before": cgroup_sample(cgroup) if cgroup else None,
        }
        for ubatch in ubatches:
            profile = profile.model_copy(update={"ubatch_size": ubatch})
            candidate = {
                "ubatch_size": ubatch,
                "requested_profile": profile.model_dump(),
                "status": "estimating",
                "trials": [],
            }
            report["candidates"].append(candidate)
            atomic_report(output, report)
            estimate = await asyncio.to_thread(
                probe_memory,
                settings,
                profile,
                backend,
                model_path,
                min(120, timeout),
                tensor_overrides=tensor_overrides,
            )
            candidate["estimate"] = asdict(estimate)
            snapshot = (await asyncio.to_thread(capture_resources, settings.work_dir)).model_dump(
                mode="json"
            )
            # Arguments use the same production runtime flags even in probe-only mode.
            pre_runtime = LlamaRuntime(settings, profile, backend)
            arguments = pre_runtime._arguments(binary or Path("llama-server"), model_path)
            arguments.extend(tensor_override_arguments(tensor_overrides))
            arguments.extend(["--verbosity", "4"])
            candidate["native_arguments"] = arguments
            candidate["preflight"] = preflight(
                estimate, snapshot, backend, arguments, cgroup_sample(cgroup) if cgroup else None
            )
            if not candidate["preflight"]["allowed"]:
                candidate["status"] = "estimated_skip"
                progress(f"ubatch {ubatch}: estimated skip")
                atomic_report(output, report)
                continue
            if probe_only:
                candidate["status"] = "estimated_safe_unmeasured"
                progress(f"ubatch {ubatch}: estimated safe (unmeasured)")
                atomic_report(output, report)
                continue
            with tempfile.TemporaryDirectory(
                prefix="pd-calibration-", dir=settings.work_dir
            ) as work:
                run_settings = settings.model_copy(update={"work_dir": Path(work)})
                assert binary is not None
                runtime = CalibrationRuntime(
                    run_settings,
                    profile,
                    backend,
                    binary=binary,
                    model_path=model_path,
                    estimate=estimate,
                    cgroup=cgroup,
                    tensor_overrides=tensor_overrides,
                )
                telemetry_path = (
                    output.resolve().parent
                    / f"{output.stem}.telemetry"
                    / (f"ubatch-{ubatch}.jsonl")
                )
                watchdog = MemoryWatchdog(runtime, telemetry_path=telemetry_path)
                recorder = watchdog.telemetry
                assert recorder is not None
                candidate["gpu_disk_telemetry"] = recorder.describe()
                candidate["cgroup_before"] = cgroup_sample(cgroup) if cgroup else None
                try:
                    await watchdog.begin()
                    candidate["native_arguments"] = runtime._arguments(binary, model_path)
                    candidate["status"] = "loading"
                    atomic_report(output, report)
                    await guarded(runtime, runtime.start(), timeout)
                    candidate["startup_ms"] = runtime.startup_ms
                    candidate["launch_admission"] = runtime.calibration_admission
                    candidate["effective_settings"] = runtime._effective_settings
                    tokens = await guarded(runtime, exact_tokens(runtime, prompt_tokens), timeout)
                    candidate["prompt_sha256"] = hashlib.sha256(
                        ",".join(map(str, tokens)).encode()
                    ).hexdigest()
                    for repeat in range(-1, repeats):
                        label = "warmup" if repeat == -1 else "measured"
                        trial = {"kind": label, "repeat": repeat, "status": "running"}
                        candidate["trials"].append(trial)
                        phase = f"{label}:{repeat}"
                        watchdog.mark_phase(phase)
                        atomic_report(output, report)
                        watchdog.phase_summary = {"sample_count": 0}
                        before = await asyncio.to_thread(watchdog.sample)
                        trial["telemetry_before"] = before
                        started = time.perf_counter()
                        try:
                            watchdog.check_sample(before)
                            trial.update(
                                await guarded(
                                    runtime, measure(runtime, tokens, output_tokens), timeout
                                )
                            )
                            trial["operation_wall_ms"] = elapsed_ms(started)
                            after = await asyncio.to_thread(watchdog.sample)
                            trial["telemetry_after"] = after
                            watchdog.check_sample(after)
                            trial["status"] = "completed"
                        except BaseException as error:
                            trial.update(
                                status="pressure"
                                if isinstance(error, PressureError)
                                else "cancelled"
                                if isinstance(error, asyncio.CancelledError)
                                else "error",
                                error=f"{type(error).__name__}: {error}",
                            )
                            raise
                        finally:
                            trial["wall_ms"] = elapsed_ms(started)
                            if "telemetry_after" not in trial:
                                # Failed requests also keep a final snapshot. Preserve
                                # their primary error if this diagnostic check fails.
                                try:
                                    after = await asyncio.to_thread(watchdog.sample)
                                    trial["telemetry_after"] = after
                                    watchdog.check_sample(after)
                                except Exception as sample_error:
                                    trial["final_sample_error"] = str(sample_error)
                                    after = watchdog.last
                            else:
                                after = trial["telemetry_after"]
                            trial["io_delta"] = {
                                key: after[key] - before[key]
                                for key in ("native_read_bytes", "native_major_faults")
                                if key in before and key in after
                            }
                            trial["peak_telemetry"] = dict(watchdog.phase_summary)
                            trial["pressure_stall_delta_us"] = {
                                f"{kind}_{key}": value
                                - before.get("cgroup", {})
                                .get("pressure", {})
                                .get(kind, {})
                                .get(key, value)
                                for kind, values in after.get("cgroup", {})
                                .get("pressure", {})
                                .items()
                                for key, value in values.items()
                            }
                            trial["gpu_disk_summary"] = recorder.summarized(
                                recorder.phases.get(phase, {})
                            )
                            candidate["gpu_disk_telemetry"] = recorder.describe()
                            trial["telemetry_cumulative"] = dict(watchdog.summary)
                            atomic_report(output, report)
                        progress(f"ubatch {ubatch}: {label} {repeat + 1} completed")
                    candidate["status"] = "measured_safe"
                except BaseException as error:
                    candidate["status"] = (
                        "pressure"
                        if isinstance(error, PressureError)
                        else "cancelled"
                        if isinstance(error, asyncio.CancelledError)
                        else "error"
                    )
                    candidate["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    # Even a failed native/client close must not strand the sampler.
                    try:
                        await stop_owned(runtime)
                    finally:
                        try:
                            await runtime.close()
                        finally:
                            await watchdog.close()
                    candidate["peak_telemetry"] = dict(watchdog.summary)
                    candidate["gpu_disk_telemetry"] = recorder.describe()
                    candidate["cgroup_after"] = cgroup_sample(cgroup) if cgroup else None
                    log = _log_tail(runtime.log_path)
                    candidate["effective_from_log"] = effective_from_log(log)
                    candidate["native_log_tail"] = log[-65536:]
                    atomic_report(output, report)
            atomic_report(output, report)
        report["status"] = "completed"
    except asyncio.CancelledError:
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["candidates"] and report["candidates"][-1]["status"] in ("estimating", "loading"):
            report["candidates"][-1].update(status="error", error=report["error"])
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["measured_safe_ubatches"] = [
            candidate["ubatch_size"]
            for candidate in report["candidates"]
            if candidate["status"] == "measured_safe"
        ]
        if cgroup:
            with contextlib.suppress(OSError, ValueError):
                report["cgroup_after"] = cgroup_sample(cgroup)
        atomic_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ubatches", type=int, nargs="+", default=list(DEFAULT_UBATCHES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--cpu-ffn", type=int)
    parser.add_argument(
        "--cpu-recurrent-projections",
        action="store_true",
        help="Opt in to mmap CPU placement for recurrent projection weights",
    )
    parser.add_argument(
        "--cpu-output-projection",
        action="store_true",
        help="Opt in to mmap CPU output.weight placement; final logits may run on CPU",
    )
    parser.add_argument("--model-id", default="qwen3.5-27b")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--cgroup", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if args.model_path:
        settings.model_path = args.model_path
    options = vars(args).copy()
    options.pop("model_path")

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await calibrate(settings, **options)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    try:
        report = asyncio.run(run())
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise SystemExit(130) from None
    print(f"{report['status']}: {args.output}")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
