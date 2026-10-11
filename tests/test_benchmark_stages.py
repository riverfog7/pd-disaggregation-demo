import math
from pathlib import Path

import pytest

from pd_disaggregation_demo.benchmarks.charts import (
    latest_complete_pair,
    phases,
    stage_comparison,
    stage_comparison_html,
)
from pd_disaggregation_demo.benchmarks.models import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSample,
)
from pd_disaggregation_demo.benchmarks.storage import save_report
from pd_disaggregation_demo.models import GenerationResult, RunMetrics


def sample(mode, *, size=256, repeat=0, order=None, success=True):
    split = mode == "split"
    return BenchmarkSample(
        prompt_tokens=size,
        repeat=repeat,
        order=int(split) if order is None else order,
        result=GenerationResult(
            mode=mode,
            model_id="qwen3.5-4b",
            prompt_tokens=size,
            generated_tokens=128,
            text="fake measured request",
            success=success,
            metrics=RunMetrics(
                tokenize_ms=2 if split else 1,
                prefill_ms=10 if split else 20,
                export_ms=3 if split else 0,
                transfer_ms=4 if split else 0,
                restore_ms=5 if split else 0,
                final_prompt_ms=6 if split else 0,
                ttft_ms=35 if split else 25,
                total_ms=85 if split else 125,
                decode_ms=77 if split else 101,
            ),
        ),
    )


def report_with_pair():
    return BenchmarkReport(
        config=BenchmarkConfig(prompt_sizes=[256], repeats=2),
        samples=[sample("local"), sample("split")],
        environment={
            "configurations": [
                {"prompt_tokens": 256, "corpus_tokenize_ms": 7.25, "startup_ms": 1000.5},
                {"prompt_tokens": 512, "corpus_tokenize_ms": 9999, "startup_ms": 9999},
            ]
        },
    )


def test_stage_rows_use_exact_pair_and_distinguish_engine_decode():
    report = report_with_pair()
    report.summary = [{"local_ttft_ms": 9999, "split_ttft_ms": 9999}]
    caption, rows, notes = stage_comparison(report)
    assert caption == "256 prompt tokens · pair 1 · local → split"
    values = {label: (local, split) for label, local, split in rows}
    assert values == {
        "Tokenization": (1, 2),
        "Prefill": (20, 10),
        "Save": (0, 3),
        "Transfer": (0, 4),
        "Restore": (0, 5),
        "Final prompt token": (0, 6),
        "Unattributed TTFT overhead": (4, 5),
        "TTFT subtotal · through first token": (25, 35),
        "After first token · client elapsed": (100, 50),
        "Total request · client elapsed": (125, 85),
        "Native decode · engine timing (not additive)": (101, 77),
    }
    assert any("outside timed pairs" in note for note in notes)
    assert notes[-1].endswith(
        "corpus tokenization 7.25 ms; local configuration/startup 1,000.50 ms."
    )
    assert not any("9,999" in note for note in notes)
    panel = stage_comparison_html(report)
    assert "Local (ms)" in panel and "Split (ms)" in panel
    assert "101.00" in panel and "77.00" in panel


def test_stage_comparison_keeps_last_pair_until_same_size_repeat_matches():
    report = report_with_pair()
    previous = latest_complete_pair(report)
    next_split = sample("split", repeat=1, order=0)
    report.samples.append(next_split)
    assert latest_complete_pair(report) == previous
    assert "retaining this pair" in stage_comparison(report)[0]
    next_local = sample("local", repeat=1, order=1)
    next_local.result.metrics.ttft_ms = 45
    report.samples.append(next_local)
    assert latest_complete_pair(report) == (next_local, next_split)
    caption, rows, _ = stage_comparison(report)
    assert caption == "256 prompt tokens · pair 2 · split → local"
    assert [row for row in rows if row[0].startswith("TTFT subtotal")] == [
        ["TTFT subtotal · through first token", 45, 35]
    ]


@pytest.mark.parametrize(
    "unmatched",
    [sample("split", size=512), sample("split", repeat=1), sample("split", success=False)],
)
def test_unmatched_requests_never_form_stage_comparison(unmatched):
    report = BenchmarkReport(
        config=BenchmarkConfig(prompt_sizes=[256]), samples=[sample("local"), unmatched]
    )
    assert latest_complete_pair(report) is None
    caption, rows, _ = stage_comparison(report)
    assert "Waiting for a complete" in caption
    assert not rows
    assert "<table" not in stage_comparison_html(report)


def test_whole_request_timeline_totals_and_first_token_boundary():
    report = report_with_pair()
    figure = phases(report)
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert [sum(trace.x[index] for trace in bars) for index in (0, 1)] == [125, 85]
    after = next(trace for trace in bars if trace.name.startswith("After first token"))
    assert list(after.x) == [100, 50]
    boundary = next(trace for trace in figure.data if trace.name == "First token / TTFT boundary")
    assert list(boundary.x) == [25, 35]
    assert not any("Native decode" in trace.name for trace in bars)
    assert "Whole-request timeline" in figure.layout.title.text


def test_incompatible_stage_timers_keep_raw_values_without_inflating_total(tmp_path):
    report = report_with_pair()
    report.samples[1].result.metrics.prefill_ms = 100
    _, rows, notes = stage_comparison(report)
    values = {label: (local, split) for label, local, split in rows}
    assert values["Prefill"] == (20, 100)
    assert values["Unattributed TTFT overhead"] == (4, None)
    assert any("instead of adding incompatible timers" in note for note in notes)
    bars = [trace for trace in phases(report).data if trace.type == "bar"]
    assert sum((trace.x[1] or 0) for trace in bars) == 85
    assert report.samples[1].result.metrics.prefill_ms == 100
    save_report(report, tmp_path)
    document = Path(report.exports["html"]).read_text()
    assert stage_comparison_html(report) in document
    assert "Whole-request timeline" in document
    assert "plotly.js" in document


def test_missing_ttft_is_unavailable_in_table_and_timeline():
    report = report_with_pair()
    report.samples[1].result.metrics.ttft_ms = None
    report.samples[1].result.metrics.decode_ms = math.nan
    _, rows, _ = stage_comparison(report)
    values = {label: (local, split) for label, local, split in rows}
    assert values["TTFT subtotal · through first token"] == (25, None)
    assert values["After first token · client elapsed"] == (100, None)
    assert values["Native decode · engine timing (not additive)"] == (101, None)
    boundary = next(trace for trace in phases(report).data if trace.type == "scatter")
    assert list(boundary.x) == [25, None]
