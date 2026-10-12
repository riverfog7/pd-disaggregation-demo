"""Upstream no-allocation estimates, separate from live memory measurements."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import Settings
from ..models import Backend, RuntimeProfile
from .arguments import model_context_arguments
from .build import ensure_memory_probe

MiB = 1024**2
_ROW = re.compile(r"^(Host|[A-Za-z][A-Za-z0-9_.:-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


def tensor_override_arguments(tensor_overrides: tuple[str, ...]) -> list[str]:
    """Explicit estimator/calibration overrides; ordinary workers do not opt in."""
    return [
        argument for override in tensor_overrides for argument in ("--override-tensor", override)
    ]


@dataclass(frozen=True)
class MemoryComponents:
    """Conservative component bounds; native output truncates each value to MiB."""

    model_bytes: int
    context_bytes: int
    compute_bytes: int
    reported_mib: tuple[int, int, int]

    @property
    def total_bytes(self) -> int:
        return self.model_bytes + self.context_bytes + self.compute_bytes


@dataclass(frozen=True)
class MemoryEstimate:
    host: MemoryComponents
    devices: dict[str, MemoryComponents]
    arguments: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""


def parse_memory_estimate(stdout: str, stderr: str = "") -> MemoryEstimate:
    """Parse the pinned helper's stdout, rejecting ambiguous/incomplete estimates."""
    rows: dict[str, MemoryComponents] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        match = _ROW.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Malformed native memory estimate row: {line!r}")
        name = match[1]
        if name in rows:
            raise ValueError(f"Duplicate native memory estimate device: {name}")
        reported = (int(match[2]), int(match[3]), int(match[4]))
        # common/fit.cpp divides integer bytes by MiB before printing. Add one
        # MiB even for zero, which can describe a nonzero sub-MiB allocation.
        rows[name] = MemoryComponents(
            model_bytes=(reported[0] + 1) * MiB,
            context_bytes=(reported[1] + 1) * MiB,
            compute_bytes=(reported[2] + 1) * MiB,
            reported_mib=reported,
        )
    if "Host" not in rows:
        raise ValueError("Native memory estimate has no Host row")
    host = rows.pop("Host")
    return MemoryEstimate(host=host, devices=rows, stdout=stdout, stderr=stderr)


def estimate_memory(
    binary: Path,
    profile: RuntimeProfile,
    backend: Backend,
    model_path: Path,
    timeout: float = 120,
    *,
    tensor_overrides: tuple[str, ...] = (),
) -> MemoryEstimate:
    """Use an existing helper; read model metadata without inference or weight allocation."""
    arguments = [
        str(binary),
        *model_context_arguments(profile, backend, model_path),
        *tensor_override_arguments(tensor_overrides),
        "--fit-print",
        "on",
    ]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=True)
    estimate = parse_memory_estimate(result.stdout, result.stderr)
    expected = {"metal": "MTL0", "cuda": "CUDA0", "cpu": None}[backend]
    if expected is not None and expected not in estimate.devices:
        raise ValueError(f"Native memory estimate has no requested device {expected}")
    return MemoryEstimate(
        host=estimate.host,
        devices=estimate.devices,
        arguments=tuple(arguments),
        stdout=result.stdout,
        stderr=result.stderr,
    )


def probe_memory(
    settings: Settings,
    profile: RuntimeProfile,
    backend: Backend,
    model_path: Path,
    timeout: float = 120,
    *,
    tensor_overrides: tuple[str, ...] = (),
) -> MemoryEstimate:
    return estimate_memory(
        ensure_memory_probe(settings, backend),
        profile,
        backend,
        model_path,
        timeout,
        tensor_overrides=tensor_overrides,
    )
