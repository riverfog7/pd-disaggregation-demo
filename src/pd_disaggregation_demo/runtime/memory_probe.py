"""No-allocation reserved-graph estimates, separate from live memory measurements.

Runtime chunk shapes can grow scheduler buffers beyond the reserved graph.
Calibration must retain live reserves and its watchdog; this is not a guarantee
of a process's eventual allocation peak.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import ENGINE_REVISION, Settings
from ..models import Backend, RuntimeProfile
from .arguments import model_context_arguments
from .build import (
    SERVER_PROBE_KIND,
    SERVER_PROBE_NAME,
    ensure_memory_probe,
    server_probe_source_sha256,
)

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
    provenance: dict = field(default_factory=dict)


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
    ]
    server_probe = binary.name == SERVER_PROBE_NAME
    if server_probe:
        arguments.extend(["--spec-type", "none"])
    arguments.extend(["--fit-print", "on"])
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=True)
    estimate = parse_memory_estimate(result.stdout, result.stderr)
    expected = {"metal": "MTL0", "cuda": "CUDA0", "cpu": None}[backend]
    if expected is not None and expected not in estimate.devices:
        raise ValueError(f"Native memory estimate has no requested device {expected}")
    provenance = {
        "kind": "generic-fit-params" if binary.name == "llama-fit-params" else "unverified",
        "binary": str(binary),
    }
    if server_probe:
        rows = [
            line.removeprefix("PD_SERVER_FIT_PARAMS ")
            for line in result.stderr.splitlines()
            if line.startswith("PD_SERVER_FIT_PARAMS ")
        ]
        if len(rows) != 1:
            raise ValueError("Server memory probe must emit exactly one provenance record")
        provenance = json.loads(rows[0])
        if not isinstance(provenance, dict):
            raise ValueError("Server memory probe provenance must be an object")
        required = {
            "kind": SERVER_PROBE_KIND,
            "engine_revision": ENGINE_REVISION,
            "probe_source_sha256": server_probe_source_sha256(),
            "parallel": 1,
            "n_outputs_max": 1,
            "n_outputs_max_per_seq": 1,
            "n_rs_seq": 0,
            "n_ctx": profile.context_size,
            "n_batch": profile.batch_size,
            "n_ubatch": profile.ubatch_size,
        }
        if any(provenance.get(key) != value for key, value in required.items()):
            raise ValueError(
                "Server memory probe provenance/context does not match requested profile"
            )
        if not str(provenance.get("linked_commit", "")).startswith(ENGINE_REVISION[:7]):
            raise ValueError("Server memory probe linked libraries do not report pinned revision")
        provenance["binary"] = str(binary)
        provenance["estimate_scope"] = "No-allocation server scheduler reserved graph"
        provenance["runtime_resize_caveat"] = (
            "Runtime chunk shapes can resize compute buffers beyond this estimate; "
            "driver cushions, live reserves and watchdog checks remain required"
        )
    return MemoryEstimate(
        host=estimate.host,
        devices=estimate.devices,
        arguments=tuple(arguments),
        stdout=result.stdout,
        stderr=result.stderr,
        provenance=provenance,
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
