"""Acceptance pass/fail semantics without starting a native engine."""

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from pd_disaggregation_demo.artifacts import validate_artifact
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import (
    ArtifactManifest,
    GenerationEvent,
    GenerationResult,
    PrefillArtifact,
    RunMetrics,
    RuntimeFingerprint,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_handoff.py"
spec = importlib.util.spec_from_file_location("handoff_validation", SCRIPT)
assert spec is not None and spec.loader is not None
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def make_artifact(settings, profile, token_ids, backend):
    manifest = ArtifactManifest(
        model_id=profile.model_id,
        fingerprint=RuntimeFingerprint(
            engine_revision=settings.engine_revision,
            model_sha256="a" * 64,
            cache_type_k=profile.cache_type,
            cache_type_v=profile.cache_type,
            context_size=profile.context_size,
        ),
        token_ids=token_ids,
        prefix_tokens=len(token_ids) - 1,
        state_bytes=4,
        state_sha256=hashlib.sha256(b"slot").hexdigest(),
        source_backend=backend,
    )
    directory = settings.artifacts_dir / manifest.artifact_id
    directory.mkdir(parents=True)
    state = directory / "state.bin"
    state.write_bytes(b"slot")
    return PrefillArtifact(manifest=manifest, state_path=state)


@pytest.fixture
def fake_engines(monkeypatch):
    scenario: dict[str, Any] = {"failure": None, "recovery_case": None, "instances": []}

    class Runtime:
        startup_ms = 1

        def __init__(self, settings, profile, backend):
            assert backend == "metal"
            self.settings = settings
            self.profile = profile
            self.closed = False
            self.restores = {}
            self.rejected_case = None
            scenario["instances"].append(self)

        async def start(self):
            pass

        async def close(self):
            self.closed = True

        async def format_chat(self, messages):
            return "formatted prompt"

        async def tokenize(self, prompt):
            return [1, 2, 3]

        async def prefill(self, token_ids):
            artifact = make_artifact(self.settings, self.profile, token_ids, "metal")
            self.fingerprint = artifact.manifest.fingerprint
            return artifact

        def baseline(self, backend="metal"):
            offset = 10 if self.profile.cache_type == "f16" else 20
            if backend == "cuda":
                offset += 20
            return f"{backend}-{self.profile.cache_type}", [offset, offset + 1]

        async def events(self, text, token_ids, restored):
            yield GenerationEvent(kind="token", data={"token_ids": token_ids})
            yield GenerationEvent(
                kind="result",
                result=GenerationResult(
                    mode="local",
                    model_id=self.profile.model_id,
                    prompt_tokens=3,
                    generated_tokens=len(token_ids),
                    text=text,
                    metrics=RunMetrics(cache_n=2 if restored else 0, prompt_n=1 if restored else 3),
                ),
            )

        def generate(self, request):
            return self.events(*self.baseline(), restored=False)

        async def decode(self, artifact, request):
            try:
                validate_artifact(artifact, self.fingerprint)
            except ValueError:
                self.rejected_case = artifact.state_path.stem
                raise
            backend = artifact.manifest.source_backend
            count = self.restores.get(artifact.manifest.artifact_id, 0) + 1
            self.restores[artifact.manifest.artifact_id] = count
            text, token_ids = self.baseline(backend)
            failure = scenario["failure"]
            if backend == "metal":
                if failure == "local-text" and count == 1:
                    text = "different text"
                if failure == "local-tokens" and count == 1:
                    token_ids = [99]
                if failure == "local-repeat" and count == 2:
                    token_ids = [99]
            elif failure == "remote-repeat" and count == 2:
                token_ids = [99]
            if self.rejected_case == scenario["recovery_case"] and self.rejected_case:
                text = "wrong recovery"
            self.rejected_case = None
            async for event in self.events(text, token_ids, restored=True):
                yield event

    class Remote:
        def __init__(self, settings):
            self.settings = settings

        async def configure_matching(self, profile):
            self.profile = profile
            return profile

        async def prefill(self, token_ids):
            self.artifact = make_artifact(self.settings, self.profile, token_ids, "cuda")
            return self.artifact.manifest

        async def download(self, manifest, fingerprint):
            return self.artifact, 2

        async def delete(self, artifact_id):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(validation, "LlamaRuntime", Runtime)
    monkeypatch.setattr(validation, "RemoteWorker", Remote)
    return scenario


async def test_acceptance_requires_consistency_but_allows_cross_device_and_cache_differences(
    tmp_path, fake_engines
):
    report = await validation.validate(Settings(work_dir=tmp_path), "qwen3.5-0.8b", 2, True)
    assert report["success"]
    assert not report["cross_cache_comparison"]["same_tokens"]
    for record in report["local"] + report["cuda_to_metal"]:
        assert record["repeat_comparison"]["same_text"]
        assert record["repeat_comparison"]["same_tokens"]
        assert {item["case"] for item in record["rejections"]} == {
            "corrupt",
            "truncated",
            "missing",
            "expired",
            "incompatible",
        }
        assert all(item["comparison"]["same_tokens"] for item in record["rejections"])
        assert all(item["comparison"]["same_text"] for item in record["rejections"])
    for record in report["cuda_to_metal"]:
        assert not record["restored_twice"][0]["comparison_with_metal_direct"]["same_tokens"]
    assert all(runtime.closed for runtime in fake_engines["instances"])


@pytest.mark.parametrize(
    "failure, message",
    [
        ("local-text", "local restore 1"),
        ("local-tokens", "local restore 1"),
        ("local-repeat", "local restore 2"),
        ("remote-repeat", "repeated CUDA artifact Metal restores"),
    ],
)
async def test_acceptance_fails_on_inconsistent_continuation(
    tmp_path, fake_engines, failure, message
):
    fake_engines["failure"] = failure
    report = await validation.validate(Settings(work_dir=tmp_path), "qwen3.5-0.8b", 2, True)
    assert not report["success"]
    assert report["error"]["type"] == "AssertionError"
    assert message in report["error"]["message"]
    assert report["local"]  # Preserve the comparisons that caused acceptance to fail.
    assert all(runtime.closed for runtime in fake_engines["instances"])
    assert not list(tmp_path.rglob("state.bin"))


@pytest.mark.parametrize("case", ["corrupt", "truncated", "missing", "expired", "incompatible"])
async def test_artifact_rejection_recovery_must_match_saved_prefix(tmp_path, fake_engines, case):
    fake_engines["recovery_case"] = case
    report = await validation.validate(Settings(work_dir=tmp_path), "qwen3.5-0.8b", 2, False)
    assert not report["success"]
    assert f"Recovery after {case} artifact rejection" in report["error"]["message"]
    assert all(runtime.closed for runtime in fake_engines["instances"])
    assert not list(tmp_path.rglob("negative-tests"))
