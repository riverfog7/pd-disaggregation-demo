from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from ..models import GenerationResult, RuntimeProfile, Schema


class BenchmarkConfig(Schema):
    profile: RuntimeProfile = Field(default_factory=RuntimeProfile)
    prompt_sizes: list[int] = Field(default_factory=lambda: [1024, 4096, 8192, 16384, 32768])
    output_tokens: int = Field(default=128, ge=1, le=8192)
    repeats: int = Field(default=5, ge=1, le=100)
    warmup: bool = True
    seed: int = 0
    bootstrap_samples: int = Field(default=10000, ge=100, le=100000)
    label: str = ""

    @model_validator(mode="after")
    def validate_sizes(self):
        if not self.prompt_sizes or any(size < 2 for size in self.prompt_sizes):
            raise ValueError("Prompt sizes must contain positive sizes of at least two tokens")
        if len(set(self.prompt_sizes)) != len(self.prompt_sizes):
            raise ValueError("Prompt sizes must be unique")
        return self


class BenchmarkSample(Schema):
    prompt_tokens: int
    repeat: int
    order: int
    result: GenerationResult


class SkippedPoint(Schema):
    prompt_tokens: int
    reason: str


class BenchmarkReport(Schema):
    schema_version: Literal[1] = 1
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: Literal["running", "completed", "cancelled", "failed"] = "running"
    config: BenchmarkConfig
    environment: dict[str, Any] = Field(default_factory=dict)
    samples: list[BenchmarkSample] = Field(default_factory=list)
    warmups: list[GenerationResult] = Field(default_factory=list)
    skipped: list[SkippedPoint] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    summary: list[dict[str, Any]] = Field(default_factory=list)
    exports: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(
        default_factory=lambda: [
            "Weights are warmed separately; measured requests start with an empty prompt cache.",
            "TTFT includes prefill, save, transfer, restore, and final prompt evaluation.",
            "Repeats alternate local/split and split/local using identical canonical tokens.",
            "Five repeats do not support a reliable tail-latency (p95) estimate.",
            "A supported TTFT win requires the lower 95% paired bootstrap interval to exceed 1.",
        ]
    )


class BenchmarkEvent(Schema):
    kind: Literal["progress", "stage", "sample", "skipped", "complete", "cancelled", "error"]
    stage: str = ""
    prompt_tokens: int | None = None
    repeat: int | None = None
    mode: Literal["local", "split"] | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    report: BenchmarkReport | None = None
