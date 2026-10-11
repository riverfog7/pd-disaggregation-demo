"""Pinned GGUF profiles shared by both machines."""

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import ModelSpec


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


def ensure_model(settings: Settings, model_id: str) -> Path:
    """Download the exact revision, then verify bytes before inference."""
    spec = get_model(model_id)
    if settings.model_path is not None:
        return verify_model(settings.model_path.expanduser(), spec)
    destination = settings.models_dir / spec.model_id
    destination.mkdir(parents=True, exist_ok=True)
    existing = destination / spec.filename
    if existing.is_file():
        return verify_model(existing, spec)
    required = spec.size_bytes + settings.disk_reserve_bytes
    available = shutil.disk_usage(destination).free
    if available < required:
        raise ValueError(f"Model download needs {required:,} disk bytes; {available:,} available")
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            revision=spec.hf_revision,
            local_dir=destination,
        )
    )
    return verify_model(downloaded, spec)
