from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

Backend = Literal["metal", "cuda", "cpu"]
Mode = Literal["local", "split"]
CacheType = Literal["f16", "q8_0"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(Schema):
    model_id: str
    hf_repo: str
    hf_revision: str
    filename: str
    sha256: str
    size_bytes: int
    layers: int
    attention_layers: int
    kv_heads: int
    head_dim: int = 256
    recurrent_bytes: int


class RuntimeProfile(Schema):
    model_id: str = "qwen3.5-4b"
    cache_type: CacheType = "f16"
    context_size: int = Field(default=8192, ge=256, le=262144)
    batch_size: int = Field(default=512, ge=1)
    ubatch_size: int = Field(default=128, ge=1)
    gpu_layers: int = Field(default=-1, ge=-1)
    cpu_ffn: int = Field(default=0, ge=0)
    threads: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_batches(self):
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size must not exceed batch_size")
        return self


class RuntimeFingerprint(Schema):
    engine_revision: str
    model_sha256: str
    cache_type_k: CacheType
    cache_type_v: CacheType
    context_size: int
    flash_attention: bool = True
    rope_settings: dict[str, Any] = Field(default_factory=dict)
    state_format_version: int = 1


class ResourceSnapshot(Schema):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    platform: str
    cpu_count: int
    cpu_model: str = "unknown"
    ram_total_bytes: int
    ram_available_bytes: int
    swap_used_bytes: int = 0
    disk_free_bytes: int
    gpu_name: str | None = None
    gpu_total_bytes: int | None = None
    gpu_free_bytes: int | None = None
    gpu_used_bytes: int | None = None
    process_rss_bytes: int = 0
    metal_recommended_bytes: int | None = None


class ArtifactManifest(Schema):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    model_id: str
    fingerprint: RuntimeFingerprint
    token_ids: list[int] = Field(min_length=2)
    prefix_tokens: int = Field(ge=1)
    state_filename: str = "state.bin"
    state_bytes: int = Field(ge=1)
    state_sha256: str
    source_backend: Backend
    timings: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prefix_boundary(self):
        if self.prefix_tokens != len(self.token_ids) - 1:
            raise ValueError("Artifact must contain exactly N-1 prompt tokens")
        if self.state_filename != "state.bin":
            raise ValueError("Artifact state filename must be state.bin")
        return self


class PrefillArtifact(Schema):
    manifest: ArtifactManifest
    state_path: Path


class PrefillRequest(Schema):
    prompt: str | None = None
    token_ids: list[int] | None = None
    backend: Literal["local", "remote"] = "local"


class GenerationRequest(Schema):
    prompt: str = ""
    token_ids: list[int] | None = None
    mode: Mode = "local"
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0)
    seed: int = 0
    ignore_eos: bool = False


class DecodeRequest(Schema):
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0)
    seed: int = 0
    ignore_eos: bool = False


class RunMetrics(Schema):
    ttft_ms: float | None = None
    tokenize_ms: float = 0
    prefill_ms: float = 0
    export_ms: float = 0
    transfer_ms: float = 0
    restore_ms: float = 0
    final_prompt_ms: float = 0
    decode_ms: float = 0
    total_ms: float = 0
    prefill_tps: float | None = None
    decode_tps: float | None = None
    client_decode_tps: float | None = None
    state_bytes: int = 0
    transfer_mbps: float | None = None
    cache_n: int = 0
    prompt_n: int = 0
    memory: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(Schema):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    mode: Mode
    model_id: str
    prompt_tokens: int
    generated_tokens: int
    text: str
    metrics: RunMetrics
    fingerprint: RuntimeFingerprint | None = None
    local_profile: RuntimeProfile | None = None
    remote_profile: RuntimeProfile | None = None
    success: bool = True
    error: str | None = None


class GenerationEvent(Schema):
    kind: Literal["stage", "token", "result", "error"]
    stage: str = ""
    text: str = ""
    token_id: int | None = None
    elapsed_ms: float = 0
    data: dict[str, Any] = Field(default_factory=dict)
    result: GenerationResult | None = None
