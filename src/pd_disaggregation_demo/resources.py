"""Live resource snapshots and conservative pre-load admission estimates."""

import ctypes
import importlib
import os
import platform
import shutil
import subprocess
from functools import cache
from pathlib import Path

import psutil
from pydantic import Field

from .config.settings import Settings
from .models import Backend, ModelSpec, ResourceSnapshot, RuntimeProfile, Schema

GiB = 1024**3
MiB = 1024**2


class AdmissionResult(Schema):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    budgets: dict[str, int] = Field(default_factory=dict)
    resources: ResourceSnapshot


def _cgroup_available(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Respect the worker's cgroup v2 limit, independently of host availability."""
    try:
        maximum = (root / "memory.max").read_text().strip()
        if maximum == "max":
            return None
        return max(0, int(maximum) - int((root / "memory.current").read_text()))
    except (OSError, ValueError):
        return None


def _gpu_snapshot() -> dict:
    try:
        pynvml = importlib.import_module("pynvml")

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            return dict(
                gpu_name=name.decode() if isinstance(name, bytes) else name,
                gpu_total_bytes=memory.total,
                gpu_free_bytes=memory.free,
                gpu_used_bytes=memory.used,
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # The container may have nvidia-smi but no loadable NVML Python binding.
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=name,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            name, total, free, used = [part.strip() for part in output.split(",")]
            return dict(
                gpu_name=name,
                gpu_total_bytes=int(total) * MiB,
                gpu_free_bytes=int(free) * MiB,
                gpu_used_bytes=int(used) * MiB,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}


def _query_metal_device() -> dict:
    """Read a real Metal device without treating unified RAM as dedicated VRAM."""
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    unsigned = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        return {}
    try:
        name = pointer(device, objc.sel_registerName(b"name"))
        utf8 = pointer(name, objc.sel_registerName(b"UTF8String")) if name else None
        recommended = unsigned(device, objc.sel_registerName(b"recommendedMaxWorkingSetSize"))
        result = {}
        if utf8:
            result["gpu_name"] = ctypes.string_at(utf8).decode("utf-8")
        if recommended > 0:
            result["metal_recommended_bytes"] = recommended
        return result
    finally:
        pointer(device, objc.sel_registerName(b"release"))


@cache
def _mac_metadata() -> dict:
    """Static hardware discovery happens once, away from the sampling hot path."""
    values = {"cpu_model": platform.processor() or "unknown"}
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        if brand:
            values["cpu_model"] = brand
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        values.update(_query_metal_device())
    except (OSError, AttributeError, ValueError):
        # Unavailable measurements remain unknown; never infer dedicated GPU memory.
        pass
    return values


def capture_resources(path: Path) -> ResourceSnapshot:
    memory = psutil.virtual_memory()
    available = memory.available
    disk_path = Path(path).resolve()
    while not disk_path.exists():
        disk_path = disk_path.parent
    own = psutil.Process(os.getpid())
    rss = own.memory_info().rss
    for process in own.children(recursive=True):
        try:
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    cpu_model = platform.processor() or "unknown"
    if platform.system() == "Linux":
        try:
            cpu_model = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except (OSError, StopIteration):
            pass
    gpu = _gpu_snapshot() if platform.system() == "Linux" else {}
    if platform.system() == "Darwin":
        metadata = _mac_metadata()
        cpu_model = metadata["cpu_model"]
        gpu = {key: value for key, value in metadata.items() if key != "cpu_model"}
    return ResourceSnapshot(
        platform=platform.system().lower(),
        cpu_count=psutil.cpu_count() or 1,
        cpu_model=cpu_model,
        ram_total_bytes=memory.total,
        ram_available_bytes=available,
        swap_used_bytes=psutil.swap_memory().used,
        disk_free_bytes=shutil.disk_usage(disk_path).free,
        process_rss_bytes=rss,
        **gpu,
    )


def check_admission(
    settings: Settings, profile: RuntimeProfile, model: ModelSpec, backend: Backend
) -> AdmissionResult:
    """Reserve headroom for weights, native compute, live KV and exported state.

    This intentionally overestimates hybrid recurrent models. CPU FFN placement has
    no architecture-specific tensor sizes in ModelSpec, so its host weight budget
    uses all weights and never claims speculative VRAM savings.
    """
    resources = capture_resources(settings.work_dir)
    cache_element_bytes = 2.0 if profile.cache_type == "f16" else 34 / 32
    kv = int(
        2
        * model.attention_layers
        * model.kv_heads
        * model.head_dim
        * profile.context_size
        * cache_element_bytes
    )
    state = kv + model.recurrent_bytes
    weights = (model.size_bytes * 11 + 9) // 10
    compute = 768 * MiB + profile.ubatch_size * model.layers * model.head_dim * 16
    if backend == "cpu":
        gpu_weights = 0
    elif profile.gpu_layers == -1:
        gpu_weights = weights
    else:
        gpu_weights = (
            weights * min(profile.gpu_layers, model.layers) + model.layers - 1
        ) // model.layers
    host_weights = weights - gpu_weights
    if profile.cpu_ffn:
        host_weights = weights
    # mmap read-only pages can be reclaimed, but startup may fault all weights.
    # Account that peak explicitly instead of assuming available swap is safe.
    ram = max(weights, host_weights + state + compute) + state + 256 * MiB
    gpu = gpu_weights + state + compute if backend == "cuda" else 0
    if backend == "metal":
        ram = weights + 2 * state + compute + 256 * MiB
        # Whole-model residency is conservative even with CPU FFN/layer placement.
        gpu = weights + state + compute
    ram_reserve = (
        max(settings.local_ram_reserve_bytes, 6 * GiB)
        if backend == "metal"
        else max(settings.remote_ram_reserve_bytes, 2 * GiB)
    )
    gpu_reserve = max(settings.gpu_reserve_bytes, GiB)
    disk_reserve = max(settings.disk_reserve_bytes, 20 * GiB)
    # Two native states cover atomic export and a currently served artifact.
    disk = model.size_bytes + 2 * state
    budgets = dict(
        weights_bytes=weights,
        kv_bytes=kv,
        state_bytes=state,
        compute_bytes=compute,
        ram_required_bytes=ram,
        gpu_required_bytes=gpu,
        disk_required_bytes=disk,
        ram_reserve_bytes=ram_reserve,
        gpu_reserve_bytes=gpu_reserve,
        disk_reserve_bytes=disk_reserve,
    )
    reasons = []
    cgroup_available = _cgroup_available() if resources.platform == "linux" else None
    if cgroup_available is not None:
        budgets["cgroup_available_bytes"] = cgroup_available
        if ram > cgroup_available:
            reasons.append(
                f"Worker cgroup has {cgroup_available:,} bytes available; "
                f"load requires {ram:,} bytes without using swap"
            )
    if ram + ram_reserve > resources.ram_available_bytes:
        reasons.append(
            f"RAM requires {ram + ram_reserve:,} bytes including reserve; "
            f"{resources.ram_available_bytes:,} available (swap is not capacity)"
        )
    if disk + disk_reserve > resources.disk_free_bytes:
        reasons.append(
            f"Disk requires {disk + disk_reserve:,} bytes including reserve; "
            f"{resources.disk_free_bytes:,} free"
        )
    if backend == "metal":
        recommended = resources.metal_recommended_bytes
        if recommended is None:
            reasons.append(
                "Metal working-set recommendation could not be measured; refusing admission"
            )
        else:
            budgets["metal_recommended_bytes"] = recommended
            if gpu + gpu_reserve > recommended:
                reasons.append(
                    f"Metal working set requires {gpu + gpu_reserve:,} bytes including reserve; "
                    f"device recommends at most {recommended:,} bytes"
                )
    if backend == "cuda":
        if resources.gpu_free_bytes is None:
            reasons.append("CUDA GPU free memory could not be measured; refusing admission")
        elif gpu + gpu_reserve > resources.gpu_free_bytes:
            reasons.append(
                f"VRAM requires {gpu + gpu_reserve:,} bytes including reserve; "
                f"{resources.gpu_free_bytes:,} free"
            )
    return AdmissionResult(
        allowed=not reasons, reasons=reasons, budgets=budgets, resources=resources
    )
