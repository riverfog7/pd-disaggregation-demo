"""Model/context options shared by the native worker and no-allocation estimator."""

from pathlib import Path

from ..models import Backend, RuntimeProfile


def model_context_arguments(
    profile: RuntimeProfile, backend: Backend, model_path: Path
) -> list[str]:
    return [
        "--model",
        str(model_path),
        "--parallel",
        "1",
        "--ctx-size",
        str(profile.context_size),
        "--batch-size",
        str(profile.batch_size),
        "--ubatch-size",
        str(profile.ubatch_size),
        "--threads",
        str(profile.threads),
        "--threads-batch",
        str(profile.threads),
        "--gpu-layers",
        str(0 if backend == "cpu" else profile.gpu_layers),
        "--n-cpu-ffn",
        str(profile.cpu_ffn),
        "--cache-type-k",
        profile.cache_type,
        "--cache-type-v",
        profile.cache_type,
        "--flash-attn",
        "on",
        "--fit",
        "off",
        "--op-offload",
        "--no-repack",
        "--no-host",
        "--load-mode",
        "mmap",
        "--device",
        {"metal": "MTL0", "cuda": "CUDA0", "cpu": "none"}[backend],
    ]
