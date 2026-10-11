"""Pinned GGUF profiles shared by both machines."""

import fcntl
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_url

from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import ModelSpec

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_ATTEMPTS = 5
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _recurrent(layers: int, value_heads: int) -> int:
    return layers * (value_heads * 128 * 128 * 4 + (2 * 16 * 128 + value_heads * 128) * 3 * 4)


_PROFILES = [
    (
        "0.8B",
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
        532517120,
        "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
        24,
        6,
        2,
        16,
    ),
    (
        "4B",
        "e87f176479d0855a907a41277aca2f8ee7a09523",
        2740937888,
        "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        32,
        8,
        4,
        32,
    ),
    (
        "9B",
        "3885219b6810b007914f3a7950a8d1b469d598a5",
        5680522464,
        "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        32,
        8,
        4,
        32,
    ),
    (
        "27B",
        "3221f178a6b842d04f1fb42f1c413534adcc0a6a",
        16740812704,
        "84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806",
        64,
        16,
        4,
        48,
    ),
]

MODELS = {
    f"qwen3.5-{size.lower()}": ModelSpec(
        model_id=f"qwen3.5-{size.lower()}",
        hf_repo=f"unsloth/Qwen3.5-{size}-GGUF",
        hf_revision=revision,
        filename=f"Qwen3.5-{size}-Q4_K_M.gguf",
        sha256=sha,
        size_bytes=byte_count,
        layers=layers,
        attention_layers=attention,
        kv_heads=kv_heads,
        recurrent_bytes=_recurrent(layers - attention, value_heads),
    )
    for size, revision, byte_count, sha, layers, attention, kv_heads, value_heads in _PROFILES
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id.lower()].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_id!r}; choose from {', '.join(MODELS)}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, spec: ModelSpec) -> Path:
    if not path.is_file() or path.stat().st_size != spec.size_bytes:
        raise ValueError(f"Model {path} must be exactly {spec.size_bytes} bytes")
    if sha256_file(path) != spec.sha256:
        raise ValueError(f"Model checksum mismatch: {path}; expected {spec.sha256}")
    return path.resolve()


def _http_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=15, read=60, write=15, pool=15),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )


def _http_download(settings: Settings, spec: ModelSpec, destination: Path) -> Path:
    """Bounded plain HTTP transfer; Xet is never invoked, even if already imported."""
    final = destination / spec.filename
    partial = final.with_name(final.name + ".part")
    if partial.is_symlink():
        raise ValueError("Model partial download must be a regular file")
    url = hf_hub_url(spec.hf_repo, spec.filename, revision=spec.hf_revision)
    reserve = max(settings.disk_reserve_bytes, 20 * 1024**3)
    with _http_client() as client:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > spec.size_bytes:
                partial.unlink()
                raise ValueError(
                    "Partial model exceeds the pinned file size; removed invalid partial"
                )
            if offset == spec.size_bytes:
                try:
                    verify_model(partial, spec)
                except ValueError:
                    partial.unlink()
                    raise
                partial.replace(final)
                return final.resolve()
            required = spec.size_bytes - offset + reserve
            available = shutil.disk_usage(destination).free
            if available < required:
                raise ValueError(
                    f"Model download needs {required:,} disk bytes; {available:,} available"
                )
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise ValueError("Model HTTP response must use identity content encoding")
                    if response.status_code == 206:
                        match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
                        if not match:
                            raise ValueError("Model resume response lacks a valid Content-Range")
                        start, end, total = map(int, match.groups())
                        if start != offset or total != spec.size_bytes or not start <= end < total:
                            raise ValueError(
                                "Model resume Content-Range does not match the pinned file"
                            )
                        expected_response_bytes = end - start + 1
                        mode = "ab"
                    elif response.status_code == 200:
                        # A server ignoring Range sends the whole file; safely restart.
                        offset = 0
                        expected_response_bytes = spec.size_bytes
                        mode = "wb"
                    else:
                        raise ValueError(f"Unexpected model download status {response.status_code}")
                    length = response.headers.get("content-length")
                    if length is not None and int(length) != expected_response_bytes:
                        raise ValueError("Model HTTP Content-Length does not match the pinned size")
                    print(
                        f"{spec.model_id}: HTTP download {offset:,}/{spec.size_bytes:,} bytes",
                        file=sys.stderr,
                        flush=True,
                    )
                    received = 0
                    next_report = offset + 64 * DOWNLOAD_CHUNK_BYTES
                    with partial.open(mode) as output:
                        for chunk in response.iter_raw(chunk_size=DOWNLOAD_CHUNK_BYTES):
                            if received + len(chunk) > expected_response_bytes:
                                raise ValueError("Model HTTP body exceeds its pinned response size")
                            if shutil.disk_usage(destination).free < reserve + len(chunk):
                                raise ValueError(
                                    "Model download would cross the 20GiB disk reserve"
                                )
                            output.write(chunk)
                            received += len(chunk)
                            if offset + received >= next_report:
                                print(
                                    f"{spec.model_id}: "
                                    f"{offset + received:,}/{spec.size_bytes:,} bytes",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                next_report = offset + received + 64 * DOWNLOAD_CHUNK_BYTES
                        output.flush()
                    if received != expected_response_bytes:
                        raise httpx.ReadError("Model HTTP body ended before the declared size")
                if partial.stat().st_size == spec.size_bytes:
                    try:
                        verify_model(partial, spec)
                    except ValueError:
                        partial.unlink()
                        raise
                    partial.replace(final)
                    return final.resolve()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code not in (
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                time.sleep(min(2**attempt, 5))
    raise RuntimeError("Model transfer did not complete after bounded retries; partial preserved")


def ensure_model(settings: Settings, model_id: str) -> Path:
    """Download pinned public weights in bounded chunks, verify and atomically publish."""
    spec = get_model(model_id)
    if settings.model_path is not None:
        return verify_model(settings.model_path.expanduser(), spec)
    destination = settings.models_dir / spec.model_id
    destination.mkdir(parents=True, exist_ok=True)
    existing = destination / spec.filename
    if existing.is_file():
        return verify_model(existing, spec)
    lock_path = destination / ("." + spec.filename + ".download.lock")
    if lock_path.is_symlink():
        raise ValueError("Model download lock must be a regular file")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if existing.is_file():
            return verify_model(existing, spec)
        return _http_download(settings, spec, destination)
