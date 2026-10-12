"""Build the pinned upstream server and metadata-only memory estimator."""

import hashlib
import json
import subprocess
from pathlib import Path

from ..config.settings import ENGINE_REVISION, Settings
from ..models import Backend

SERVER_PROBE_NAME = "pd-server-fit-params"
SERVER_PROBE_KIND = "server-context-v1"
SERVER_PROBE_SOURCE = Path(__file__).parent / "native" / "server_fit"


def server_probe_source_sha256() -> str:
    return hashlib.sha256((SERVER_PROBE_SOURCE / "server-fit-params.cpp").read_bytes()).hexdigest()


def _run(arguments: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def ensure_engine(settings: Settings, backend: Backend) -> Path:
    if settings.engine_revision != ENGINE_REVISION:
        raise ValueError("Only the pinned llama.cpp engine revision is supported")
    if settings.engine_binary:
        binary = settings.engine_binary.resolve()
        version = _run([str(binary), "--version"])
        if ENGINE_REVISION[:7] not in version:
            raise ValueError("Configured llama-server does not report the pinned revision")
        return binary
    root = settings.cache_dir / "engines" / ENGINE_REVISION
    source = root / "source"
    build = root / "build" / backend
    binary = build / "bin" / "llama-server"
    stamp = build / "pd-build.json"
    expected = {"revision": ENGINE_REVISION, "backend": backend}
    if binary.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == expected:
        helper = binary.with_name("llama-fit-params")
        if not helper.is_file():
            _run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "llama-fit-params",
                    "-j",
                    str(settings.build_jobs),
                ]
            )
        return binary
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        _run(["git", "init", str(source)])
        _run(["git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp"], source)
    revision = (
        _run(["git", "rev-parse", "HEAD"], source).strip()
        if (source / "CMakeLists.txt").exists()
        else ""
    )
    if revision != ENGINE_REVISION:
        _run(["git", "fetch", "--depth", "1", "origin", ENGINE_REVISION], source)
        _run(["git", "checkout", "--detach", ENGINE_REVISION], source)
    if _run(["git", "rev-parse", "HEAD"], source).strip() != ENGINE_REVISION:
        raise RuntimeError("Invalid upstream source revision")
    flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_CURL=OFF",
        f"-DCMAKE_PROJECT_INCLUDE={SERVER_PROBE_SOURCE / 'inject.cmake'}",
        f"-DGGML_METAL={'ON' if backend == 'metal' else 'OFF'}",
        f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
    ]
    if backend == "cuda":
        flags.append("-DCMAKE_CUDA_ARCHITECTURES=75")
    _run(["cmake", "-S", str(source), "-B", str(build), *flags])
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "llama-server",
            "llama-fit-params",
            SERVER_PROBE_NAME,
            "-j",
            str(settings.build_jobs),
        ]
    )
    stamp.write_text(json.dumps(expected))
    return binary


def ensure_memory_probe(settings: Settings, backend: Backend) -> Path:
    """Require the explicit server-context probe, retaining the generic fit tool separately."""
    helper = ensure_engine(settings, backend).with_name(SERVER_PROBE_NAME)
    expected_sha = server_probe_source_sha256()
    if helper.is_file():
        version = _run([str(helper), "--version"])
        if (
            f"engine={ENGINE_REVISION}" in version
            and SERVER_PROBE_KIND in version
            and f"source_sha256={expected_sha}" in version
            and f"linked_commit={ENGINE_REVISION[:7]}" in version
        ):
            return helper
    if settings.engine_binary:
        if helper.is_file():
            raise ValueError(
                "Configured server memory probe does not report pinned revision, "
                "server context kind and current source hash; rebuild the probe"
            )
        raise FileNotFoundError(
            f"Pinned {SERVER_PROBE_NAME} must be installed beside server: {helper}; "
            "generic llama-fit-params has different output graph limits"
        )
    root = settings.cache_dir / "engines" / ENGINE_REVISION
    build = root / "build" / backend
    _run(
        [
            "cmake",
            "-S",
            str(root / "source"),
            "-B",
            str(build),
            f"-DCMAKE_PROJECT_INCLUDE={SERVER_PROBE_SOURCE / 'inject.cmake'}",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            SERVER_PROBE_NAME,
            "-j",
            str(settings.build_jobs),
        ]
    )
    if not helper.is_file():
        raise FileNotFoundError(f"Server memory probe build did not publish {helper}")
    version = _run([str(helper), "--version"])
    if (
        f"engine={ENGINE_REVISION}" not in version
        or SERVER_PROBE_KIND not in version
        or f"source_sha256={expected_sha}" not in version
        or f"linked_commit={ENGINE_REVISION[:7]}" not in version
    ):
        raise ValueError(
            "Built server memory probe does not report pinned revision and source hash"
        )
    return helper
