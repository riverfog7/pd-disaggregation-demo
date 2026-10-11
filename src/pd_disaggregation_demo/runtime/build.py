"""Build only the pinned upstream llama-server, with explicit backend selection."""

import json
import subprocess
from pathlib import Path

from ..config.settings import ENGINE_REVISION, Settings
from ..models import Backend


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
        f"-DGGML_METAL={'ON' if backend == 'metal' else 'OFF'}",
        f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
    ]
    if backend == "cuda":
        flags.append("-DCMAKE_CUDA_ARCHITECTURES=75")
    _run(["cmake", "-S", str(source), "-B", str(build), *flags])
    _run(
        ["cmake", "--build", str(build), "--target", "llama-server", "-j", str(settings.build_jobs)]
    )
    stamp.write_text(json.dumps(expected))
    return binary
