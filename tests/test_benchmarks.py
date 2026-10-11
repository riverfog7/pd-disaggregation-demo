import asyncio
import csv
from pathlib import Path

import pytest

from pd_disaggregation_demo.benchmarks import BenchmarkConfig, collect_benchmark, load_report
from pd_disaggregation_demo.benchmarks.stats import paired_speedup
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import (
    GenerationEvent,
    GenerationResult,
    RunMetrics,
    RuntimeProfile,
)


class FakePipeline:
    def __init__(self):
        self.profile = RuntimeProfile()
        self.calls = []
        self.configurations = []
        self.cancelled = False
        self.reject = None
        self.bad_count = False
        self.cancel_signal = None

    async def health(self):
        return {
            "status": "fake",
            "profile": self.profile.model_dump(),
            "resources": {"gpu": "fake"},
            "git_revision": "test-only",
        }

    async def configure(self, profile):
        if self.reject == profile.context_size:
            raise ValueError("insufficient memory reserve")
        self.profile = profile
        self.configurations.append(profile)

    async def tokenize(self, text):
        return [ord(char) for char in text]

    async def cancel(self):
        self.cancelled = True

    async def generate(self, request):
        self.calls.append(request)
        if self.cancel_signal:
            self.cancel_signal.set()
        yield GenerationEvent(kind="stage", stage="prefill")
        size = len(request.token_ids or [])
        split = request.mode == "split"
        yield GenerationEvent(
            kind="result",
            result=GenerationResult(
                mode=request.mode,
                model_id=self.profile.model_id,
                prompt_tokens=size,
                generated_tokens=request.max_tokens - int(self.bad_count),
                text="measured fake",
                local_profile=self.profile,
                metrics=RunMetrics(
                    ttft_ms=50 if split else 100,
                    prefill_ms=25 if split else 90,
                    export_ms=5 if split else 0,
                    transfer_ms=10 if split else 0,
                    restore_ms=5 if split else 0,
                    final_prompt_ms=2 if split else 0,
                    decode_ms=100,
                    total_ms=150 if split else 200,
                    prefill_tps=100 if split else 50,
                    decode_tps=10,
                    state_bytes=10000 if split else 0,
                    transfer_mbps=1 if split else None,
                    cache_n=size - 1 if split else 0,
                    prompt_n=1 if split else size,
                    memory={"local_rss_bytes": 12345},
                ),
            ),
        )


def settings(tmp_path):
    return Settings(work_dir=tmp_path, cache_dir=tmp_path / "cache")


async def test_paired_exact_tokens_warmup_context_and_exports(tmp_path):
    pipeline = FakePipeline()
    config = BenchmarkConfig(
        prompt_sizes=[256, 512],
        output_tokens=4,
        repeats=3,
        profile=RuntimeProfile(context_size=256),
        bootstrap_samples=100,
    )
    report = await collect_benchmark(pipeline, config, settings(tmp_path))
    assert report.status == "completed"
    assert len(report.samples) == 12
    assert len(report.warmups) == 4
    assert [p.context_size for p in pipeline.configurations] == [260, 516]
    for size in config.prompt_sizes:
        samples = [s for s in report.samples if s.prompt_tokens == size]
        assert [s.result.mode for s in samples] == [
            "local",
            "split",
            "split",
            "local",
            "local",
            "split",
        ]
        requests = [r for r in pipeline.calls if len(r.token_ids) == size]
        assert all(r.token_ids == requests[0].token_ids for r in requests)
        assert all(r.ignore_eos and r.max_tokens == 4 for r in requests)
    assert report.summary[0]["speedup"] == pytest.approx(2)
    assert report.summary[0]["ci_low"] == pytest.approx(2)
    assert report.summary[0]["supported_win"]
    loaded = load_report(Path(report.exports["json"]))
    assert loaded == report
    with Path(report.exports["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert rows[1]["transfer_ms"] == "10.0"
    assert '"local_rss_bytes": 12345' in rows[0]["memory"]
    document = Path(report.exports["html"]).read_text()
    assert "plotly.js" in document
    assert '<script src="https://cdn.plot.ly' not in document
    assert "test-only" in document


async def test_unsafe_point_skipped_no_fake_result(tmp_path):
    pipeline = FakePipeline()
    pipeline.reject = 260
    config = BenchmarkConfig(
        prompt_sizes=[256, 512, 262144],
        output_tokens=4,
        repeats=1,
        profile=RuntimeProfile(context_size=256),
        bootstrap_samples=100,
    )
    report = await collect_benchmark(pipeline, config, settings(tmp_path))
    assert report.status == "completed"
    assert [s.prompt_tokens for s in report.skipped] == [256, 262144]
    assert [s.prompt_tokens for s in report.samples] == [512, 512]
    assert report.summary[0]["speedup"] is None


async def test_count_mismatch_fails_without_fallback(tmp_path):
    pipeline = FakePipeline()
    pipeline.bad_count = True
    config = BenchmarkConfig(prompt_sizes=[256], repeats=1, warmup=False, bootstrap_samples=100)
    report = await collect_benchmark(pipeline, config, settings(tmp_path))
    assert report.status == "failed"
    assert "Token budget mismatch" in report.errors[0]
    assert not report.samples
    assert report.failures[0]["result"]["generated_tokens"] == 127
    assert len(pipeline.calls) == 1
    assert Path(report.exports["json"]).exists()


async def test_cancel_active_request_saves_partial_report(tmp_path):
    pipeline = FakePipeline()
    signal = asyncio.Event()
    pipeline.cancel_signal = signal
    config = BenchmarkConfig(prompt_sizes=[256], repeats=2, warmup=False, bootstrap_samples=100)
    report = await collect_benchmark(pipeline, config, settings(tmp_path), cancel_event=signal)
    assert report.status == "cancelled"
    assert pipeline.cancelled
    assert not report.samples
    assert load_report(Path(report.exports["json"])).status == "cancelled"


def test_paired_bootstrap_uncertain_and_single_pair():
    assert not paired_speedup([100], [10], seed=0, draws=100)["supported_win"]
    uncertain = paired_speedup([10, 100], [100, 10], seed=0, draws=1000)
    assert uncertain["ci_low"] < 1 < uncertain["ci_high"]
    assert not uncertain["supported_win"]
    with pytest.raises(ValueError):
        paired_speedup([0], [10], seed=0, draws=100)


def test_ui_build_and_mount_smoke(tmp_path):
    gr = pytest.importorskip("gradio")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pd_disaggregation_demo.ui.app import build_ui

    demo = build_ui(FakePipeline(), settings(tmp_path))
    app = gr.mount_gradio_app(FastAPI(), demo, path="/demo")
    with TestClient(app) as client:
        response = client.get("/demo/config")
        assert response.status_code == 200
        config = response.json()
        labels = [c["props"].get("label") for c in config["components"]]
        assert "Prompt sizes (tokens)" in labels
        assert "Download JSON · CSV · standalone HTML" in labels
        values = [c["props"].get("value") for c in config["components"]]
        assert "Run paired benchmark" in values
        assert "qwen3.5-4b" in values


async def test_worker_unreachable_is_failure_not_unsafe_skip(tmp_path):
    class Unreachable(FakePipeline):
        async def configure(self, profile):
            raise ConnectionError("worker unreachable")

    report = await collect_benchmark(
        Unreachable(), BenchmarkConfig(prompt_sizes=[256]), settings(tmp_path)
    )
    assert report.status == "failed"
    assert not report.skipped
    assert report.errors == ["worker unreachable"]


async def test_split_admission_skips_and_continues(tmp_path):
    class AdmissionError(RuntimeError):
        pass

    class RemoteUnsafe(FakePipeline):
        async def generate(self, request):
            if request.mode == "split" and self.profile.context_size == 260:
                raise AdmissionError("requested allocation rejected")
            async for event in super().generate(request):
                yield event

    report = await collect_benchmark(
        RemoteUnsafe(),
        BenchmarkConfig(
            profile=RuntimeProfile(context_size=256),
            prompt_sizes=[256, 512],
            output_tokens=4,
            repeats=1,
            bootstrap_samples=100,
        ),
        settings(tmp_path),
    )
    assert report.status == "completed"
    assert [s.prompt_tokens for s in report.skipped] == [256]
    assert [s.prompt_tokens for s in report.samples] == [512, 512]


async def test_ui_async_callback_outputs_and_gradio6_history(tmp_path):
    pytest.importorskip("gradio")
    from pd_disaggregation_demo.ui.app import _text_history, build_ui

    assert _text_history([{"role": "user", "content": [{"type": "text", "text": "hello"}]}]) == [
        {"role": "user", "content": "hello"}
    ]
    pipeline = FakePipeline()
    demo = build_ui(pipeline, settings(tmp_path))
    fn = next(fn for fn in demo.fns.values() if fn.name == "benchmark")
    updates = []
    async for update in fn.fn("qwen3.5-4b", "f16", 256, 512, 128, "256", 4, 1, "test"):
        assert len(update) == len(fn.outputs)
        updates.append(update)
    assert "completed" in updates[-1][0]
    assert len(updates[-1][9]) == 3
