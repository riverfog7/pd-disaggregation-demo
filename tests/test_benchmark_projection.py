import csv
from pathlib import Path

import pytest

from pd_disaggregation_demo.benchmarks.charts import curves, projection_html
from pd_disaggregation_demo.benchmarks.models import (
    ONE_GBPS_NOTE,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSample,
)
from pd_disaggregation_demo.benchmarks.stats import (
    add_modeled_projection,
    modeled_split_ttft_ms,
    modeled_transfer_ms,
    summarize,
)
from pd_disaggregation_demo.benchmarks.storage import list_reports, load_report, save_report
from pd_disaggregation_demo.models import GenerationResult, RunMetrics


def sample(mode, repeat=0):
    split = mode == "split"
    return BenchmarkSample(
        prompt_tokens=256,
        repeat=repeat,
        order=int(split),
        result=GenerationResult(
            mode=mode,
            model_id="qwen3.5-4b",
            prompt_tokens=256,
            generated_tokens=128,
            text="fake measured result",
            metrics=RunMetrics(
                ttft_ms=400 if split else 200,
                transfer_ms=350 if split else 0,
                state_bytes=12_500_000 if split else 0,
                total_ms=500,
                export_ms=12 if split else 0,
                restore_ms=8 if split else 0,
            ),
        ),
    )


def paired_report(repeats=3):
    return BenchmarkReport(
        config=BenchmarkConfig(prompt_sizes=[256], repeats=repeats, bootstrap_samples=100),
        samples=[sample(mode, repeat) for repeat in range(repeats) for mode in ("local", "split")],
    )


def test_ideal_projection_keeps_all_other_observed_overhead():
    split = sample("split").result
    before = split.model_dump()
    assert modeled_transfer_ms(split) == 100
    assert modeled_split_ttft_ms(split) == 150
    assert split.model_dump() == before
    # Faster measured transfers can produce a slower projected TTFT; never clamp to a win.
    split.metrics.transfer_ms = 50
    assert modeled_split_ttft_ms(split) == 450
    split.metrics.transfer_ms = split.metrics.ttft_ms
    assert modeled_split_ttft_ms(split) == 100
    assert modeled_split_ttft_ms(sample("local").result) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("transfer_ms", 401),
        ("transfer_ms", -1),
        ("transfer_ms", float("nan")),
        ("transfer_ms", float("inf")),
        ("ttft_ms", None),
        ("ttft_ms", 0),
        ("ttft_ms", float("nan")),
        ("ttft_ms", float("inf")),
        ("state_bytes", 0),
        ("state_bytes", -1),
    ],
)
def test_invalid_physical_inputs_do_not_create_projected_ttft(field, value):
    split = sample("split").result
    setattr(split.metrics, field, value)
    assert modeled_split_ttft_ms(split) is None
    split.success = False
    assert modeled_split_ttft_ms(split) is None


def test_modeled_pair_ci_never_replaces_measured_win_or_uses_unmatched_pairs():
    report = paired_report()
    report.samples.append(sample("local", repeat=3))
    row = summarize(report)[0]
    assert row["speedup"] == pytest.approx(0.5)
    assert row["ci_high"] == pytest.approx(0.5)
    assert not row["supported_win"]
    assert row["modeled_1gbps_split_ttft_ms"] == 150
    assert row["modeled_1gbps_speedup"] == pytest.approx(4 / 3)
    assert row["modeled_1gbps_ci_low"] == pytest.approx(4 / 3)
    assert row["modeled_1gbps_pairs"] == 3
    assert not any("supported_win" in key for key in row if key.startswith("modeled"))
    report.samples[1].result.metrics.transfer_ms = 401
    row = summarize(report)[0]
    assert row["pairs"] == 3
    assert row["modeled_1gbps_pairs"] == 2
    one_pair = summarize(paired_report(repeats=1))[0]
    assert one_pair["modeled_1gbps_ci_low"] is None
    assert one_pair["modeled_1gbps_ci_high"] is None


def test_legacy_reports_derive_model_without_changing_files_or_measured_summary(tmp_path):
    report = paired_report()
    report.summary = summarize(report)
    legacy = {
        key: value for key, value in report.summary[0].items() if not key.startswith("modeled")
    }
    report.summary = [legacy.copy()]
    report.notes.remove(ONE_GBPS_NOTE)
    path = tmp_path / f"{report.run_id}.json"
    path.write_text(report.model_dump_json())
    original = path.read_bytes()
    loaded = load_report(path)
    assert {key: loaded.summary[0][key] for key in legacy} == legacy
    assert loaded.summary[0]["modeled_1gbps_split_ttft_ms"] == 150
    assert loaded.notes.count(ONE_GBPS_NOTE) == 1
    add_modeled_projection(loaded)
    assert loaded.notes.count(ONE_GBPS_NOTE) == 1
    assert list_reports(tmp_path)[0].summary == loaded.summary
    assert path.read_bytes() == original
    # Direct rendering of an old in-memory report works as well as history loading.
    figure = curves([report])
    model = [trace for trace in figure.data if trace.name == "1Gbps model"]
    assert list(model[0].y) == [150]
    assert list(model[1].y) == pytest.approx([4 / 3])
    assert all(trace.line.dash == "dash" for trace in model)
    assert not any(
        trace.name == "1Gbps model" for trace in figure.data if trace.yaxis in ("y3", "y4")
    )
    assert report.summary[0]["supported_win"] == legacy["supported_win"]


def test_model_exports_remain_labeled_and_raw_timers_unchanged(tmp_path):
    report = paired_report()
    report.summary = summarize(report)
    before = [item.result.metrics.model_dump() for item in report.samples]
    save_report(report, tmp_path)
    assert load_report(Path(report.exports["json"])) == report
    with Path(report.exports["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["modeled_1gbps_split_ttft_ms"] == ""
    assert float(rows[1]["ttft_ms"]) == 400
    assert float(rows[1]["transfer_ms"]) == 350
    assert float(rows[1]["modeled_1gbps_transfer_ms"]) == 100
    assert float(rows[1]["modeled_1gbps_split_ttft_ms"]) == 150
    assert "not measured" in rows[1]["modeled_1gbps_assumption"]
    document = Path(report.exports["html"]).read_text()
    assert projection_html(report) in document
    assert "optimistic projection" in document
    assert "not a LAN measurement" in document
    assert before == [item.result.metrics.model_dump() for item in report.samples]
