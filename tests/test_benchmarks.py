import asyncio
import contextlib
import csv
import json
import re
import time
from pathlib import Path

import pytest

from pd_disaggregation_demo.benchmarks import BenchmarkConfig, collect_benchmark, load_report
from pd_disaggregation_demo.benchmarks.runner import CORPUS, exact_tokens, run_benchmark
from pd_disaggregation_demo.benchmarks.stats import paired_speedup
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import (
    GenerationEvent,
    GenerationRequest,
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
        self.session_owner = None
        self.formatted_messages = None

    @contextlib.asynccontextmanager
    async def session(self):
        if self.session_owner is not None:
            raise RuntimeError("The demo supports one active request")
        self.session_owner = asyncio.current_task()
        try:
            yield
        finally:
            self.session_owner = None

    async def format_chat(self, messages):
        self.formatted_messages = messages
        return "native template prompt"

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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_paired_bootstrap_rejects_nonfinite_timing(invalid):
    with pytest.raises(ValueError, match="finite"):
        paired_speedup([100, invalid], [50, 50], seed=0, draws=100)


@pytest.mark.parametrize(
    "ttft,total,message",
    [
        (None, 200, "measured TTFT"),
        (0, 200, "measured TTFT"),
        (float("nan"), 200, "measured TTFT"),
        (float("inf"), 200, "measured TTFT"),
        (100, float("nan"), "total duration"),
        (100, float("inf"), "total duration"),
        (100, 99, "total duration"),
    ],
)
async def test_invalid_native_timing_fails_without_estimation(tmp_path, ttft, total, message):
    class InvalidTiming(FakePipeline):
        async def generate(self, request):
            async for event in super().generate(request):
                if event.result is not None:
                    event.result.metrics.ttft_ms = ttft
                    event.result.metrics.total_ms = total
                yield event

    report = await collect_benchmark(
        InvalidTiming(),
        BenchmarkConfig(prompt_sizes=[256], repeats=1, warmup=False, bootstrap_samples=100),
        settings(tmp_path),
    )
    assert report.status == "failed"
    assert message in report.errors[0]
    assert not report.samples
    assert report.summary[0]["speedup"] is None
    assert not report.summary[0]["supported_win"]
    assert load_report(Path(report.exports["json"])).status == "failed"


async def test_partial_pair_failure_uses_only_complete_pairs(tmp_path):
    class LastSplitFails(FakePipeline):
        async def generate(self, request):
            if len(self.calls) == 5:
                raise RuntimeError("last split request failed")
            async for event in super().generate(request):
                yield event

    report = await collect_benchmark(
        LastSplitFails(),
        BenchmarkConfig(prompt_sizes=[256], repeats=3, warmup=False, bootstrap_samples=100),
        settings(tmp_path),
    )
    assert report.status == "failed"
    assert len(report.samples) == 5
    row = report.summary[0]
    assert (row["local_samples"], row["split_samples"], row["pairs"]) == (3, 2, 2)
    assert row["speedup"] == pytest.approx(2)
    assert report.failures[0]["mode"] == "split"
    assert report.failures[0]["repeat"] == 2


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
    assert "Waiting for a complete" in updates[0][11]
    assert "256 prompt tokens · pair 1" in updates[-1][11]
    assert "Native decode · engine timing (not additive)" in updates[-1][11]
    assert updates[-1][11] in Path(updates[-1][9][-1]).read_text()
    assert "Modeled ideal 1Gbps" in updates[-1][12]
    assert "not a LAN measurement" in updates[-1][12]


async def test_history_downloads_use_stable_api_routes_under_demo_mount(tmp_path):
    gr = pytest.importorskip("gradio")
    import httpx

    from pd_disaggregation_demo.api import create_app
    from pd_disaggregation_demo.ui.app import build_ui

    pipeline = FakePipeline()
    runtime_settings = settings(tmp_path)
    report = await collect_benchmark(
        pipeline,
        BenchmarkConfig(prompt_sizes=[256], repeats=1, label="<saved run>", bootstrap_samples=100),
        runtime_settings,
    )
    demo = build_ui(pipeline, runtime_settings)
    compare = next(fn for fn in demo.fns.values() if fn.name == "compare")
    _, markup = compare.fn([report.run_id])
    assert "&lt;saved run&gt;" in markup
    urls = re.findall(r'href="([^"]+)"', markup)
    assert urls == [
        f"/reports/{report.run_id}/{extension}" for extension in ("json", "csv", "html")
    ]
    assert "gradio_api/file" not in markup
    api = create_app(runtime_settings, pipeline=pipeline, with_ui=False)
    api = gr.mount_gradio_app(api, demo, path="/demo")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://test"
    ) as client:
        assert (await client.get("/demo/config")).status_code == 200
        for url, extension in zip(urls, ("json", "csv", "html"), strict=True):
            response = await client.get(url)
            assert response.status_code == 200
            assert response.content == Path(report.exports[extension]).read_bytes()


async def test_ui_closing_callback_saves_cancelled_report_and_releases_session(tmp_path):
    pytest.importorskip("gradio")
    from pd_disaggregation_demo.ui.app import build_ui

    pipeline = FakePipeline()
    demo = build_ui(pipeline, settings(tmp_path))
    fn = next(fn for fn in demo.fns.values() if fn.name == "benchmark")
    stream = fn.fn("qwen3.5-4b", "f16", 256, 512, 128, "256", 4, 1, "test")
    update = await anext(stream)
    assert not update[-1]["interactive"]
    assert pipeline.session_owner is asyncio.current_task()
    await stream.aclose()
    assert pipeline.session_owner is None
    assert pipeline.cancelled
    report = load_report(next(settings(tmp_path).reports_dir.glob("*.json")))
    assert report.status == "cancelled"
    assert not report.samples


async def test_ui_export_failure_enables_run_and_preserves_measurements(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from pd_disaggregation_demo.ui.app import build_ui

    def cannot_save(report, directory):
        raise OSError("report storage unavailable")

    monkeypatch.setattr("pd_disaggregation_demo.benchmarks.runner.save_report", cannot_save)
    pipeline = FakePipeline()
    demo = build_ui(pipeline, settings(tmp_path))
    fn = next(fn for fn in demo.fns.values() if fn.name == "benchmark")
    updates = [
        update async for update in fn.fn("qwen3.5-4b", "f16", 256, 512, 128, "256", 4, 1, "test")
    ]
    assert all(len(update) == len(fn.outputs) for update in updates)
    assert "Failed" in updates[-1][0]
    assert "report storage unavailable" in updates[-1][0]
    assert updates[-1][-1]["interactive"]
    assert updates[-1][9] == {"__type__": "update"}
    assert len(updates[-2][8]["samples"]) == 2
    assert updates[-2][9] is None
    assert pipeline.session_owner is None


async def test_ui_close_storage_failure_releases_session_without_yielding(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from pd_disaggregation_demo.ui.app import build_ui

    def cannot_save(report, directory):
        raise OSError("report storage unavailable")

    monkeypatch.setattr("pd_disaggregation_demo.benchmarks.runner.save_report", cannot_save)
    pipeline = FakePipeline()
    demo = build_ui(pipeline, settings(tmp_path))
    fn = next(fn for fn in demo.fns.values() if fn.name == "benchmark")
    stream = fn.fn("qwen3.5-4b", "f16", 256, 512, 128, "256", 4, 1, "test")
    await anext(stream)
    with pytest.raises(OSError, match="report storage unavailable"):
        await stream.aclose()
    assert pipeline.session_owner is None
    assert pipeline.cancelled


async def test_token_preparation_is_bounded_and_exact():
    class Tokenizer:
        def __init__(self):
            self.lengths = []

        async def tokenize(self, text):
            self.lengths.append(len(text))
            return [len(word) for word in text.split()]

    tokenizer = Tokenizer()
    tokens = await exact_tokens(tokenizer, 32768)
    assert len(tokens) == 32768
    assert len(tokenizer.lengths) <= 3
    assert max(tokenizer.lengths) < 32768 * 10
    with pytest.raises(ValueError, match="between"):
        await exact_tokens(tokenizer, 10**12)

    class Broken:
        async def tokenize(self, text):
            assert len(text) <= len(CORPUS)
            return []

    with pytest.raises(ValueError, match="too few"):
        await exact_tokens(Broken(), 32768)


async def test_benchmark_keeps_session_while_yielding_progress(tmp_path):
    pipeline = FakePipeline()
    stream = run_benchmark(
        pipeline, BenchmarkConfig(prompt_sizes=[256], repeats=1), settings(tmp_path)
    )
    event = await anext(stream)
    assert event.stage == "prepare"
    assert pipeline.session_owner is asyncio.current_task()

    async def competing_request():
        async with pipeline.session():
            raise AssertionError("Concurrent API/chat entered an active benchmark session")

    with pytest.raises(RuntimeError, match="one active request"):
        await asyncio.create_task(competing_request())
    await stream.aclose()
    assert pipeline.session_owner is None
    assert pipeline.cancelled
    assert load_report(next(settings(tmp_path).reports_dir.glob("*.json"))).status == "cancelled"


async def test_chat_uses_native_template_in_same_session(tmp_path):
    pytest.importorskip("gradio")
    from pd_disaggregation_demo.ui.app import build_ui

    pipeline = FakePipeline()
    demo = build_ui(pipeline, settings(tmp_path))
    fn = next(fn for fn in demo.fns.values() if fn.name == "chat")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "prior question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "prior answer"}]},
    ]
    async for _ in fn.fn("new question", messages, "local", "qwen3.5-4b", "f16", 256, 512, 128, 4):
        assert pipeline.session_owner is asyncio.current_task()
    assert pipeline.calls[0].prompt == "native template prompt"
    assert pipeline.formatted_messages == [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "new question"},
    ]
    assert pipeline.session_owner is None


async def test_local_cached_request_is_not_a_valid_benchmark(tmp_path):
    class UnexpectedCache(FakePipeline):
        async def generate(self, request):
            async for event in super().generate(request):
                if event.result:
                    event.result.metrics.cache_n = 1
                yield event

    report = await collect_benchmark(
        UnexpectedCache(),
        BenchmarkConfig(
            prompt_sizes=[256],
            warmup=False,
            repeats=1,
        ),
        settings(tmp_path),
    )
    assert report.status == "failed"
    assert "Local boundary mismatch" in report.errors[0]
    assert not report.samples
    assert any("Sampled peaks can miss short transients" in note for note in report.notes)


async def test_real_pipeline_reserves_benchmark_against_api_requests(tmp_path):
    from pd_disaggregation_demo.pipeline import Pipeline

    pipeline = Pipeline(settings(tmp_path))
    remote_client = pipeline.remote
    pipeline.runtime = FakePipeline()
    pipeline.remote = FakePipeline()
    stream = run_benchmark(
        pipeline, BenchmarkConfig(prompt_sizes=[256], repeats=1), settings(tmp_path)
    )
    try:
        assert (await anext(stream)).stage == "prepare"

        async def competing_api_generation():
            async for _ in pipeline.generate(GenerationRequest(token_ids=[1, 2], max_tokens=1)):
                raise AssertionError("API generation entered reserved benchmark")

        with pytest.raises(RuntimeError, match="one active request"):
            await asyncio.create_task(competing_api_generation())
        with pytest.raises(RuntimeError, match="one active request"):
            await asyncio.create_task(pipeline.configure(RuntimeProfile(context_size=256)))
        assert (await anext(stream)).stage == "configure"
        assert (await anext(stream)).stage == "warmup"
        assert pipeline._active is asyncio.current_task()
        assert pipeline._lock.locked()
    finally:
        await stream.aclose()
        await remote_client.close()
    assert not pipeline._lock.locked()
    assert pipeline._active is None


async def test_memory_sampled_peaks_and_transfer_units(tmp_path):
    from pd_disaggregation_demo.benchmarks.charts import memory_chart, transfer

    report = await collect_benchmark(
        FakePipeline(),
        BenchmarkConfig(
            prompt_sizes=[256],
            repeats=1,
            bootstrap_samples=100,
        ),
        settings(tmp_path),
    )
    report.samples[0].result.metrics.memory = {
        "before": {"process_rss_bytes": 1024**3, "disk_free_bytes": 999 * 1024**3},
        "peak": {"process_rss_max_bytes": 2 * 1024**3, "sample_count": 3, "sample_period_ms": 100},
    }
    figure = memory_chart(report)
    traces = {trace.name: trace for trace in figure.data}
    assert list(traces["peak.process_rss_max_bytes"].y) == [2]
    assert not any("disk_free" in name for name in traces)
    assert not any("gpu_used" in name for name in traces)
    assert "short transients may be missed" in figure.layout.title.text
    assert "Transfer rate (Mbit/s)" in [item.text for item in transfer(report).layout.annotations]


async def test_cancellation_closes_native_stream_before_releasing_session(tmp_path):
    class ClosingPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            self.stream_closed = False

        async def generate(self, request):
            try:
                async for event in super().generate(request):
                    yield event
            finally:
                self.stream_closed = True

        async def cancel(self):
            assert self.stream_closed
            await super().cancel()

    pipeline = ClosingPipeline()
    signal = asyncio.Event()
    pipeline.cancel_signal = signal
    report = await collect_benchmark(
        pipeline,
        BenchmarkConfig(
            prompt_sizes=[256],
            repeats=1,
            warmup=False,
        ),
        settings(tmp_path),
        cancel_event=signal,
    )
    assert report.status == "cancelled"
    assert pipeline.stream_closed
    assert pipeline.session_owner is None


async def test_slow_consumer_cannot_pause_timed_native_stages(tmp_path):
    class FakeTimedPipeline(FakePipeline):
        in_timed_request = False

        async def generate(self, request):
            self.in_timed_request = True
            start = time.perf_counter()
            try:
                yield GenerationEvent(kind="stage", stage="prefill", elapsed_ms=0)
                await asyncio.sleep(0.002)
                yield GenerationEvent(
                    kind="stage",
                    stage="restore",
                    elapsed_ms=2,
                    data={"source": "actual fake stage"},
                )
                await asyncio.sleep(0.002)
                # Measure elapsed wall time across both native stage yields. A forwarding
                # harness would include each slow consumer delay in the reported TTFT.
                ttft_ms = (time.perf_counter() - start) * 1000
                async for event in super().generate(request):
                    if event.result:
                        event.result.metrics.ttft_ms = ttft_ms
                        yield event
            finally:
                self.in_timed_request = False

    pipeline = FakeTimedPipeline()
    config = BenchmarkConfig(
        prompt_sizes=[256], output_tokens=4, repeats=1, warmup=False, bootstrap_samples=100
    )
    consumer_delay = 0.08
    consumer_observed_timed_work = []
    observed_kinds = []
    report = None
    async for event in run_benchmark(pipeline, config, settings(tmp_path)):
        observed_kinds.append(event.kind)
        consumer_observed_timed_work.append(pipeline.in_timed_request)
        report = event.report
        await asyncio.sleep(consumer_delay)
    assert report.status == "completed"
    assert len(report.samples) == 2
    assert all(sample.result.metrics.ttft_ms < consumer_delay * 1000 for sample in report.samples)
    assert not any(consumer_observed_timed_work)
    assert "stage" not in observed_kinds
    stages = report.environment["measured_stage_events"]
    assert [stage["stage"] for stage in stages] == ["prefill", "restore", "prefill", "restore"]
    assert stages[1]["elapsed_ms"] == 2
    assert stages[1]["data"] == {"source": "actual fake stage"}
    assert any("Dashboard phase updates occur after each sample" in note for note in report.notes)


async def test_csv_keeps_effective_settings_builds_and_failure_context(tmp_path):
    from pd_disaggregation_demo.benchmarks import save_report
    from pd_disaggregation_demo.benchmarks.models import SkippedPoint

    report = await collect_benchmark(
        FakePipeline(),
        BenchmarkConfig(
            prompt_sizes=[256],
            repeats=1,
            bootstrap_samples=100,
        ),
        settings(tmp_path),
    )
    effective = {
        "local": {"n_batch": 256, "n_ubatch": 64, "cache_type_k": "f16"},
        "remote": {"n_batch": 1024, "gpu_layers": 33},
    }
    report.samples[1].result.effective_settings = effective
    report.status = "failed"
    report.errors = ["worker disconnected after completed pair"]
    report.skipped = [SkippedPoint(prompt_tokens=32768, reason="context memory admission refused")]
    report.environment.update(
        git_revision="local-recorded-sha",
        engine_revision="engine-pinned-sha",
        health_before={"remote": {"git_revision": "remote-recorded-sha"}},
    )
    save_report(report, settings(tmp_path).reports_dir)
    with Path(report.exports["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(report.samples) == 2
    assert json.loads(rows[1]["effective_settings"]) == effective
    assert json.loads(rows[0]["effective_settings"]) == {}
    assert all(row["local_git_revision"] == "local-recorded-sha" for row in rows)
    assert all(row["engine_revision"] == "engine-pinned-sha" for row in rows)
    assert all(row["remote_git_revision"] == "remote-recorded-sha" for row in rows)
    assert all(row["status"] == "failed" for row in rows)
    assert json.loads(rows[1]["benchmark_errors"]) == report.errors
    assert json.loads(rows[1]["benchmark_skipped"]) == [
        point.model_dump() for point in report.skipped
    ]
    assert json.loads(rows[1]["benchmark_config"]) == report.config.model_dump(mode="json")
    assert float(rows[1]["ttft_ms"]) == report.samples[1].result.metrics.ttft_ms

    report.environment["health_before"] = {"remote": {"git_revision": "unknown"}}
    report.environment["health_after"] = {"remote": {"git_revision": "remote-after-sha"}}
    save_report(report, settings(tmp_path).reports_dir)
    with Path(report.exports["csv"]).open() as stream:
        assert next(csv.DictReader(stream))["remote_git_revision"] == "remote-after-sha"

    report.environment.clear()
    save_report(report, settings(tmp_path).reports_dir)
    with Path(report.exports["csv"]).open() as stream:
        row = next(csv.DictReader(stream))
    assert (
        row["local_git_revision"]
        == row["engine_revision"]
        == row["remote_git_revision"]
        == "unknown"
    )

    report.samples.clear()
    save_report(report, settings(tmp_path).reports_dir)
    with Path(report.exports["csv"]).open() as stream:
        reader = csv.DictReader(stream)
        assert "effective_settings" in reader.fieldnames
        assert "remote_git_revision" in reader.fieldnames
        assert not list(reader)
