import math
from pathlib import Path

import pytest

from pd_disaggregation_demo.benchmarks.charts import (
    curves,
    latest_complete_pair,
    memory_chart,
    phases,
    stage_comparison,
    stage_comparison_html,
    transfer,
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
    boundary = next(trace for trace in figure.data if trace.name == "First token / TTFT")
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


def test_charts_reserve_full_width_for_data_on_narrow_panels():
    report = report_with_pair()
    report.summary = [{"prompt_tokens": 256, "local_ttft_ms": 25, "split_ttft_ms": 35}]
    comparison, timeline = curves([report]), phases(report)
    for figure in (comparison, timeline):
        assert figure.layout.autosize
        assert figure.layout.width is None
        assert figure.layout.legend.orientation == "h"
        assert figure.layout.legend.y < 0
        assert figure.layout.legend.entrywidthmode == "fraction"
        assert figure.layout.legend.entrywidth == 0.5
        assert figure.layout.margin.b >= 190
        assert figure.layout.xaxis.tickangle == 0
    assert comparison.layout.height >= 1000
    for name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        assert list(getattr(comparison.layout, name).domain) == [0, 1]
    assert list(comparison.data[0].y) == [25]
    assert timeline.layout.height >= 480
    assert timeline.layout.title.text == "Whole-request timeline"
    assert all(len(trace.name) <= 22 for trace in timeline.data)


@pytest.mark.parametrize(
    "labels",
    [
        ("same run label", "same run label"),
        ("same-long-prefix-16-first", "same-long-prefix-16-second"),
    ],
)
def test_same_prefix_runs_have_independent_legend_groups_and_visible_ids(labels):
    reports = [report_with_pair(), report_with_pair()]
    for report, label, identifier in zip(reports, labels, ("a" * 32, "b" * 32), strict=True):
        report.run_id = identifier
        report.config.label = label
        report.summary = [{"prompt_tokens": 256, "local_ttft_ms": 25, "split_ttft_ms": 35}]
        report.samples[1].result.metrics.state_bytes = 1_000_000
    figure = curves(reports)
    for mode in ("local", "split"):
        groups = [
            trace.legendgroup
            for trace in figure.data
            if trace.showlegend and trace.name.endswith(mode)
        ]
        assert set(groups) == {f"{report.run_id}-{mode}" for report in reports}
        assert len(groups) == 2
        names = [trace.name for trace in figure.data if trace.legendgroup in groups]
        assert len(set(names)) == 2
        for report in reports:
            related = [
                trace for trace in figure.data if trace.legendgroup == f"{report.run_id}-{mode}"
            ]
            assert len(related) == 3  # Toggle only this run's TTFT, prefill and decode traces.
            assert all(report.run_id[:6] in trace.name for trace in related)
    modeled = [trace for trace in figure.data if trace.name.endswith("1Gbps model")]
    assert {trace.legendgroup for trace in modeled} == {
        f"{report.run_id}-modeled-1gbps" for report in reports
    }
    assert len({trace.name for trace in modeled}) == 2


def test_transfer_panels_stack_without_changing_units_or_missing_values():
    report = report_with_pair()
    report.summary = [
        {"prompt_tokens": 256, "split_state_bytes": 2 * 1024**2, "split_transfer_mbps": 42.5},
        {"prompt_tokens": 512, "split_state_bytes": None, "split_transfer_mbps": None},
    ]
    figure = transfer(report)
    assert list(figure.layout.xaxis.domain) == list(figure.layout.xaxis2.domain) == [0, 1]
    assert figure.data[0].yaxis == "y"
    assert figure.data[1].yaxis == "y2"
    assert list(figure.data[0].y) == [2, None]
    assert list(figure.data[1].y) == [42.5, None]
    assert figure.layout.height >= 600
    assert figure.layout.autosize and figure.layout.width is None
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.entrywidthmode == "fraction"
    assert figure.layout.legend.y < 0
    assert figure.layout.margin.b >= 160
    assert figure.layout.xaxis.tickangle == 0


def test_memory_short_legend_keeps_full_metric_identity_and_values_in_hover():
    report = report_with_pair()
    report.samples[0].result.metrics.memory = {
        "before": {"process_rss_bytes": 1024**3, "native_process_rss_bytes": 2 * 1024**3},
        "peak": {"process_rss_max_bytes": 3 * 1024**3},
        "remote_prefill_peak": {"gpu_used_max_bytes": 4 * 1024**3},
        "unusual_nested_peak_namespace": {
            "cgroup_extraordinarily_long_memory_limit_bytes": 5 * 1024**3
        },
    }
    figure = memory_chart(report)
    traces = {trace.meta["metric"]: trace for trace in figure.data}
    assert list(traces["before.process_rss_bytes"].y) == [1]
    assert list(traces["before.native_process_rss_bytes"].y) == [2]
    assert list(traces["peak.process_rss_max_bytes"].y) == [3]
    assert list(traces["remote_prefill_peak.gpu_used_max_bytes"].y) == [4]
    assert len(traces) == 5
    for full_name, trace in traces.items():
        assert trace.legendgroup == full_name
        assert "%{meta.metric}" in trace.hovertemplate
        assert "GiB" in trace.hovertemplate
        assert len(trace.name) <= 24
    assert figure.layout.autosize and figure.layout.width is None
    assert figure.layout.height >= 500
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.entrywidthmode == "fraction"
    assert figure.layout.legend.entrywidth == 0.5
    assert figure.layout.legend.maxheight == 120
    assert figure.layout.legend.y < 0
    assert figure.layout.margin.b >= 190
    assert figure.layout.xaxis.tickangle == 0
    assert "short transients may be missed" in figure.layout.title.text


@pytest.mark.parametrize("sizes", [[1024, 4096, 8192], [1024, 4096, 8192, 16384, 32768]])
def test_log_axes_label_actual_prompt_sizes_for_three_point_and_default_runs(sizes):
    report = report_with_pair()
    report.summary = [
        {"prompt_tokens": size, "local_ttft_ms": index + 1} for index, size in enumerate(sizes)
    ]
    for figure in (curves([report]), transfer(report)):
        axes = [
            value
            for key, value in figure.to_plotly_json()["layout"].items()
            if key.startswith("xaxis")
        ]
        assert axes
        for axis in axes:
            assert axis["type"] == "log"
            assert axis["tickmode"] == "array"
            assert axis["minorloglabels"] == "none"
            assert axis["tickvals"] == sizes
            assert axis["ticktext"] == [f"{size:,}" for size in sizes]
    assert list(curves([report]).data[0].x) == sizes


def test_prompt_ticks_use_comparison_union_and_thin_without_changing_data():
    first, second = report_with_pair(), report_with_pair()
    first.summary = [{"prompt_tokens": size} for size in (1024, 4096)]
    second.summary = [{"prompt_tokens": size} for size in (2048, 8192)]
    assert list(curves([first, second]).layout.xaxis.tickvals) == [1024, 2048, 4096, 8192]
    many = [1024 * n for n in range(1, 40)]
    first.summary = [{"prompt_tokens": size, "local_ttft_ms": size / 10} for size in many]
    for figure in (curves([first]), transfer(first)):
        ticks = list(figure.layout.xaxis.tickvals)
        assert 2 <= len(ticks) <= 5
        assert ticks[0] == many[0] and ticks[-1] == many[-1]
        assert set(ticks) <= set(many)
        assert list(figure.data[0].x) == many
