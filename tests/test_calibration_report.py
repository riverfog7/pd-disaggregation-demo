import csv
import json

import plotly.graph_objects as go
import pytest

from pd_disaggregation_demo.calibration_report import (
    GiB,
    MiB,
    render_calibration_reports,
    summarize_calibrations,
)


def save_report(tmp_path, *, timeline=None):
    def trial(kind, repeat, seconds, status="completed"):
        return {
            "kind": kind,
            "repeat": repeat,
            "status": status,
            "native_prompt_n": 8191,
            "native_cache_n": 0,
            "native_saved_n": 8191,
            "manifest": {
                "token_ids": [99] * 8192,
                "timings": {
                    "native_prefill_ms": seconds * 1000,
                    "prefill_ms": seconds * 1000 + 100,
                    "prefill_tps": 8191 / seconds,
                    "native_save_ms": 500,
                    "export_ms": 700,
                },
            },
            "io_delta": {"native_read_bytes": 2 * GiB, "native_major_faults": 200},
            "telemetry_after": {"gpu_total_bytes": 8 * GiB},
            "peak_telemetry": {
                "gpu_free_min_bytes": 2 * GiB,
                "ram_available_min_bytes": 4 * GiB,
                "cgroup_anon_bytes_max": GiB,
                "cgroup_file_bytes_max": 5 * GiB,
            },
        }

    path = tmp_path / "input.json"
    candidate = {
        "ubatch_size": 512,
        "status": "measured_safe",
        "startup_ms": 20000,
        "trials": [
            trial("warmup", -1, 100),
            trial("measured", 0, 10),
            trial("measured", 1, 20),
            trial("measured", 2, 999, "error"),
        ],
    }
    if timeline:
        candidate["gpu_disk_telemetry"] = timeline
    path.write_text(
        json.dumps(
            {
                "synthetic_layout_only": True,
                "backend": "cuda",
                "status": "completed",
                "placement": {"cpu_ffn": 64, "tensor_overrides": []},
                "environment": {"git_revision": "a" * 40, "image_sha256": "b" * 64},
                "candidates": [
                    candidate,
                    {
                        "ubatch_size": 8192,
                        "status": "estimated_skip",
                        "preflight": {"reasons": ["Insufficient GPU reserve"]},
                    },
                ],
            }
        )
    )
    return path


def test_only_completed_measured_trials_contribute_and_tokens_not_copied(tmp_path):
    source = save_report(tmp_path)
    report = summarize_calibrations([source])
    row = report["candidates"][0]
    assert row["completed_measured_trials"] == 2
    assert row["measured_median_native_prefill_seconds"] == 15
    assert row["startup_seconds"] == 20
    assert report["trials"][0]["kind"] == "warmup"
    assert report["trials"][1]["native_export_seconds"] == 0.5
    assert report["trials"][1]["client_export_seconds"] == 0.7
    assert report["trials"][1]["largest_exercised_prompt_chunk"] == 512
    assert report["trials"][1]["physical_read_gib"] == 2
    assert report["trials"][1]["device_vram_used_peak_gib"] == 6
    assert report["candidates"][1]["status"] == "estimated_skip"
    assert report["candidates"][1]["completed_measured_trials"] == 0
    assert "token_ids" not in json.dumps(report)
    assert report["sources"][0]["environment"]["image_sha256"] == "b" * 64


def test_timeline_relocated_file_rates_do_not_cross_phases_or_counter_resets(tmp_path):
    columns = ["elapsed_ms", "phase", "gpu_utilization_percent", "native_read_bytes"]
    path = tmp_path / "activity.jsonl"
    samples = [
        [0, "load", 10, 0],
        [1000, "load", 20, MiB],
        [2000, "measured:0", 5, 10 * MiB],
        [3000, "measured:0", 7, 12 * MiB],
        [4000, "measured:0", None, 0],
    ]
    path.write_text(
        json.dumps({"schema_version": 1, "columns": columns})
        + "\n"
        + "\n".join(json.dumps(row) for row in samples)
    )
    source = save_report(
        tmp_path,
        timeline={
            "jsonl_path": "/remote/obsolete/activity.jsonl",
            "truncated": True,
            "dropped_samples": 10,
            "phases": {
                "measured:0": {
                    "time_weighted_gpu_utilization_percent": 6,
                    "fraction_gpu_utilization_below_20_percent": 1,
                    "gpu_coverage_ms": 1000,
                }
            },
        },
    )
    report = summarize_calibrations([source])
    timeline = report["timelines"][0]
    assert timeline["available"]
    assert [row["physical_read_mib_per_second"] for row in timeline["samples"]] == [
        None,
        1,
        None,
        2,
        None,
    ]
    assert report["candidates"][0]["timeline_truncated"] is True
    assert report["trials"][1]["gpu_activity_time_weighted_percent"] == 6


def test_missing_timeline_remains_unavailable_not_zero_activity(tmp_path):
    source = save_report(tmp_path, timeline={"jsonl_path": "/absent/no-data.jsonl"})
    report = summarize_calibrations([source])
    assert not report["timelines"][0]["available"]
    assert "unavailable" in report["timelines"][0]["reason"]
    assert report["trials"][1]["gpu_activity_time_weighted_percent"] is None


def test_render_self_contained_html_csv_and_json_with_separate_phases(tmp_path):
    source = save_report(tmp_path)
    outputs = render_calibration_reports([source], tmp_path / "rendered")
    document = outputs["html"].read_text()
    assert "plotly.js" in document
    assert '<script src="https://' not in document
    assert "Low activity alone does not prove a disk bottleneck" in document
    assert "estimated_skip" in document and "warmup" in document
    assert "token_ids" not in document
    assert "synthetic_layout_only" in document
    rows = list(csv.DictReader(outputs["summary_csv"].open()))
    assert rows[0]["measured_median_native_prefill_seconds"] == "15.0"
    assert rows[1]["completed_measured_trials"] == "0"
    assert json.loads(outputs["summary_json"].read_text())["sources"][0]["metadata"][
        "synthetic_layout_only"
    ]


def test_multiple_reports_keep_placement_and_provenance_distinct(tmp_path):
    source = save_report(tmp_path)
    second = tmp_path / "second.json"
    value = json.loads(source.read_text())
    value["placement"]["tensor_overrides"] = [r"\.ssm_out\.weight$=CPU"]
    second.write_text(json.dumps(value))
    report = summarize_calibrations([source, second])
    assert len(report["sources"]) == 2
    assert report["candidates"][0]["placement"] != report["candidates"][2]["placement"]
    assert report["candidates"][2]["source_id"] == "source-2"


def test_malformed_timeline_is_rejected(tmp_path):
    timeline = tmp_path / "invalid.jsonl"
    timeline.write_text('{"columns":["elapsed_ms","phase"]}\n[1]\n')
    source = save_report(tmp_path, timeline={"jsonl_path": str(timeline)})
    with pytest.raises(ValueError, match="Malformed timeline row"):
        summarize_calibrations([source])


def test_sampler_trial_summary_and_collected_telemetry_directory(tmp_path):
    directory = tmp_path / "input.telemetry"
    directory.mkdir()
    (directory / "ubatch-512.jsonl").write_text(
        '{"columns":["elapsed_ms","phase"]}\n[0,"measured:0"]\n'
    )
    source = save_report(
        tmp_path, timeline={"jsonl_path": "/remote/input.telemetry/ubatch-512.jsonl"}
    )
    raw = json.loads(source.read_text())
    raw["candidates"][0]["trials"][1]["gpu_disk_summary"] = {
        "time_weighted_gpu_utilization_percent": 7,
        "fraction_gpu_utilization_below_20_percent": 0.95,
    }
    raw["candidates"][0]["trials"][1]["pressure_stall_delta_us"] = {
        "memory_full_total_us": 500000,
        "io_full_total_us": 1000000,
    }
    source.write_text(json.dumps(raw))
    report = summarize_calibrations([source])
    assert report["timelines"][0]["available"]
    row = report["trials"][1]
    assert row["gpu_activity_time_weighted_percent"] == 7
    assert row["gpu_activity_below_20_fraction"] == 0.95
    assert row["memory_full_stall_seconds"] == 0.5
    assert row["io_full_stall_seconds"] == 1


def test_independent_metric_charts_use_responsive_config_and_integer_ticks(tmp_path, monkeypatch):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["candidates"][0]["ubatch_size"] = 16384
    source.write_text(json.dumps(raw))
    rendered = []

    def capture(figure, **options):
        rendered.append((figure.to_plotly_json(), options))
        return "<div class='plotly-graph-div'></div>"

    monkeypatch.setattr(go.Figure, "to_html", capture)
    output = render_calibration_reports([source], tmp_path / "responsive")
    assert len(rendered) == 6
    for figure, options in rendered:
        assert "xaxis2" not in figure["layout"]
        axis = figure["layout"]["xaxis"]
        assert axis["tickmode"] == "array"
        assert axis["tickvals"] == [8192, 16384]
        assert axis["ticktext"] == ["8192", "16384"]
        assert "dtick" not in axis
        assert not figure["layout"].get("title", {}).get("text")
        assert options["config"]["responsive"] is True
    assert rendered[0][1]["include_plotlyjs"] is True
    assert all(options["include_plotlyjs"] is False for _, options in rendered[1:])
    document = output["html"].read_text()
    assert "@media(max-width:720px)" in document
    assert "grid-template-columns:minmax(0,1fr)" in document
    assert document.count("class='chart-card'") == 6


def test_measured_results_precede_collapsed_method_and_labels_are_compact(tmp_path):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["placement"].update(
        cpu_recurrent_projections=True,
        tensor_overrides=[r"^blk\.[0-9]+\.(attn_qkv|attn_gate|ssm_out)\.weight$=CPU"],
    )
    source.write_text(json.dumps(raw))
    output = render_calibration_reports([source], tmp_path / "first-results")
    document = output["html"].read_text()
    assert "CUDA: FFN64+recurrent" in document
    assert document.index("<table>") < document.index("<details>")
    assert document.index("<div class='chart-grid'>") < document.index("<details>")
    assert "<details><summary>Method and interpretation</summary><ul>" in document
    assert "GPU % (whole trial)" in document
    assert "they are not prefill-only kernel measurements" in document
    assert "<summary>All candidates, estimated skips, and failures</summary>" in document


@pytest.mark.parametrize("output_field", [True, False])
def test_recurrent_and_output_cpu_placement_are_both_named(tmp_path, output_field):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["placement"].update(
        cpu_recurrent_projections=True,
        cpu_output_projection=output_field,
        tensor_overrides=[] if output_field else [r"^output\.weight$=CPU"],
    )
    source.write_text(json.dumps(raw))
    report = summarize_calibrations([source])
    assert report["candidates"][0]["placement"] == "CUDA: FFN64+recurrent+output"


def test_mixed_prompt_lengths_have_visible_run_context_and_unambiguous_legends(
    tmp_path, monkeypatch
):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw.update(
        prompt_tokens=8192, requested_profile={"model_id": "qwen3.5-27b", "cache_type": "q8_0"}
    )
    source.write_text(json.dumps(raw))
    second = tmp_path / "16k.json"
    raw["prompt_tokens"] = 16384
    second.write_text(json.dumps(raw))
    names = []

    def capture(figure, **options):
        names.append([trace.name for trace in figure.data])
        return "<div class='plotly-graph-div'></div>"

    monkeypatch.setattr(go.Figure, "to_html", capture)
    output = render_calibration_reports([source, second], tmp_path / "mixed")
    document = output["html"].read_text()
    assert "8,192 prompt tokens" in document and "16,384 prompt tokens" in document
    assert "qwen3.5-27b" in document and "q8_0 cache" in document
    assert document.index("16,384 prompt tokens") < document.index("<table>")
    assert all(legend == ["Run 1", "Run 2"] for legend in names)
    summary = json.loads(output["summary_json"].read_text())
    assert summary["candidates"][0]["prompt_tokens"] == 8192
    assert summary["candidates"][2]["prompt_tokens"] == 16384
    assert summary["candidates"][0]["model_id"] == "qwen3.5-27b"
    assert summary["candidates"][0]["cache_type"] == "q8_0"


def test_comparable_reports_connect_sorted_points_but_never_mix_context(tmp_path, monkeypatch):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw.update(
        prompt_tokens=8192,
        requested_profile={
            "model_id": "qwen3.5-27b",
            "cache_type": "q8_0",
            "context_size": 8320,
            "threads": 4,
            "gpu_layers": -1,
            "batch_size": 8192,
        },
    )
    raw["candidates"][0]["ubatch_size"] = 3840
    source.write_text(json.dumps(raw))
    second = tmp_path / "second.json"
    raw["candidates"][0]["ubatch_size"] = 2048
    second.write_text(json.dumps(raw))
    third = tmp_path / "third.json"
    raw["requested_profile"]["context_size"] = 16512
    third.write_text(json.dumps(raw))
    figures = []
    monkeypatch.setattr(
        go.Figure,
        "to_html",
        lambda figure, **options: figures.append(figure.to_plotly_json()) or "<div></div>",
    )
    render_calibration_reports([source, second, third], tmp_path / "grouped")
    assert len(figures[0]["data"]) == 2
    connected, separate = figures[0]["data"]
    assert connected["x"] == [2048, 3840]
    assert connected["name"] == "Run 2 + Run 1"
    assert connected["customdata"] == [["Run 2", 8192], ["Run 1", 8192]]
    assert separate["x"] == [2048]
    assert separate["name"] == "Run 3"


def test_different_actual_ram_caps_and_prompt_hashes_are_separate_and_visible(
    tmp_path, monkeypatch
):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw.update(
        prompt_tokens=8192,
        requested_profile={
            "model_id": "qwen3.5-27b",
            "cache_type": "q8_0",
            "context_size": 8320,
            "threads": 4,
            "gpu_layers": -1,
            "batch_size": 8192,
        },
    )
    raw["candidates"][0].update(prompt_sha256="a" * 64, cgroup_before={"limit_bytes": 8 * GiB})
    source.write_text(json.dumps(raw))
    second = tmp_path / "more-ram.json"
    raw["candidates"][0]["cgroup_before"]["limit_bytes"] = 9 * GiB
    second.write_text(json.dumps(raw))
    third = tmp_path / "different-prompt.json"
    raw["candidates"][0]["prompt_sha256"] = "b" * 64
    third.write_text(json.dumps(raw))
    names = []
    monkeypatch.setattr(
        go.Figure,
        "to_html",
        lambda figure, **options: (
            names.append([trace.name for trace in figure.data]) or "<div></div>"
        ),
    )
    output = render_calibration_reports([source, second, third], tmp_path / "ram-cap")
    assert names[0] == ["Run 1 · 8 GiB RAM cap", "Run 2 · 9 GiB RAM cap", "Run 3 · 9 GiB RAM cap"]
    document = output["html"].read_text()
    assert "8 GiB RAM cap" in document and "9 GiB RAM cap" in document
    assert document.index("9 GiB RAM cap") < document.index("<table>")


@pytest.mark.parametrize("field,value", [("threads", 8), ("gpu_layers", 40), ("batch_size", 16384)])
def test_execution_setting_difference_prevents_joining_curves(tmp_path, monkeypatch, field, value):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw.update(
        prompt_tokens=8192,
        requested_profile={
            "model_id": "qwen3.5-27b",
            "cache_type": "q8_0",
            "context_size": 8320,
            "threads": 4,
            "gpu_layers": -1,
            "batch_size": 8192,
        },
    )
    source.write_text(json.dumps(raw))
    second = tmp_path / "different-execution.json"
    raw["candidates"][0]["requested_profile"] = {field: value}
    second.write_text(json.dumps(raw))
    figures = []
    monkeypatch.setattr(
        go.Figure,
        "to_html",
        lambda figure, **options: figures.append(figure.to_plotly_json()) or "<div></div>",
    )
    output = render_calibration_reports([source, second], tmp_path / "execution")
    assert len(figures[0]["data"]) == 2
    summary = json.loads(output["summary_json"].read_text())
    assert summary["candidates"][0][field] != summary["candidates"][2][field]


@pytest.mark.parametrize("attention_field", [True, False])
def test_attention_label_is_independent_of_recurrent_and_vocabulary_output(
    tmp_path, attention_field
):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["placement"].update(
        cpu_recurrent_projections=True,
        cpu_output_projection=True,
        cpu_attention_projections=attention_field,
        tensor_overrides=[]
        if attention_field
        else [r"^blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output)\.weight$=CPU"],
    )
    source.write_text(json.dumps(raw))
    report = summarize_calibrations([source])
    assert report["candidates"][0]["placement"] == "CUDA: FFN64+recurrent+attention+output"


def test_attention_only_placement_uses_short_label(tmp_path):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["placement"].update(
        cpu_attention_projections=True,
        tensor_overrides=[r"^blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output)\.weight$=CPU"],
    )
    source.write_text(json.dumps(raw))
    assert summarize_calibrations([source])["candidates"][0]["placement"] == "CUDA: FFN64+attention"


def test_nonzero_process_exit_visible_without_discarding_completed_timings(tmp_path):
    source = save_report(tmp_path)
    sidecar = {"container_exit_code": 139, "oom_killed": False, "image_id": "sha256:abc"}
    (tmp_path / "process-status.json").write_text(json.dumps(sidecar))
    output = render_calibration_reports([source], tmp_path / "cleanup-failed")
    summary = json.loads(output["summary_json"].read_text())
    assert summary["sources"][0]["process_status"] == sidecar
    assert summary["sources"][0]["process_status_path"] == str(tmp_path / "process-status.json")
    assert summary["candidates"][0]["completed_measured_trials"] == 2
    assert summary["candidates"][0]["measured_median_native_prefill_seconds"] == 15
    document = output["html"].read_text()
    message = "Trials recorded; process cleanup failed (exit 139)"
    assert message in document
    assert document.index(message) < document.index("<table>")
    assert "process-failure" in document


@pytest.mark.parametrize(
    "status,expected",
    [
        (None, "Process exit status unknown"),
        ({"container_exit_code": 0}, "Process exited cleanly (exit 0)"),
        ({"container_exit_code": None}, "Process exit status unknown"),
    ],
)
def test_missing_or_unmeasured_process_exit_is_not_implicitly_clean(tmp_path, status, expected):
    source = save_report(tmp_path)
    if status is not None:
        (tmp_path / "process-status.json").write_text(json.dumps(status))
    output = render_calibration_reports([source], tmp_path / "exit-status")
    summary = json.loads(output["summary_json"].read_text())
    assert summary["sources"][0]["process_status"] == status
    document = output["html"].read_text()
    assert expected in document
    if status is None:
        assert summary["sources"][0]["process_status_path"] is None
        assert "Process exited cleanly" not in document


def test_incomplete_report_nonzero_exit_not_labelled_post_trial_cleanup(tmp_path):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["status"] = "stopped"
    source.write_text(json.dumps(raw))
    (tmp_path / "process-status.json").write_text('{"container_exit_code":137}')
    document = render_calibration_reports([source], tmp_path / "stopped")["html"].read_text()
    assert "Process failed (exit 137)" in document
    assert "process cleanup failed" not in document


def test_timeline_headings_identify_run_and_prompt_for_repeated_microbatch(tmp_path):
    telemetry = tmp_path / "activity.jsonl"
    telemetry.write_text(
        '{"columns":["elapsed_ms","phase","gpu_utilization_percent"]}\n'
        '[0,"measured:0",10]\n[1000,"measured:0",20]\n'
    )
    source = save_report(tmp_path, timeline={"jsonl_path": str(telemetry)})
    raw = json.loads(source.read_text())
    raw["prompt_tokens"] = 8192
    raw["candidates"][0]["ubatch_size"] = 8192
    source.write_text(json.dumps(raw))
    second = tmp_path / "16k.json"
    raw["prompt_tokens"] = 16384
    second.write_text(json.dumps(raw))
    output = render_calibration_reports([source, second], tmp_path / "timeline-labels")
    document = output["html"].read_text()
    assert "Run 1 · CUDA: FFN64 · 8,192 prompt tokens · microbatch 8192</h3>" in document
    assert "Run 2 · CUDA: FFN64 · 16,384 prompt tokens · microbatch 8192</h3>" in document
    measured = document.split("<table>", 1)[1].split("</table>", 1)[0]
    assert "<th>Prompt tokens</th>" in measured
    assert "<td>8192</td>" in measured and "<td>16384</td>" in measured
    assert "Process exit status unknown" in document
    assert "grid-template-columns:minmax(0,1fr)" in document


@pytest.mark.parametrize(
    "report_status,code,expected",
    [
        ("cancelled", 130, "Cancelled (exit 130); incomplete trials excluded"),
        ("cancelled", 139, "Process failed (exit 139)"),
        ("stopped", 130, "Process failed (exit 130)"),
        ("completed", 130, "Trials recorded; process cleanup failed (exit 130)"),
    ],
)
def test_only_report_cancelled_and_exit130_have_neutral_cancellation_label(
    tmp_path, report_status, code, expected
):
    source = save_report(tmp_path)
    raw = json.loads(source.read_text())
    raw["status"] = report_status
    raw["candidates"][0]["trials"][-1]["status"] = "cancelled"
    source.write_text(json.dumps(raw))
    sidecar = {"container_exit_code": code, "reason": "operator redirection"}
    (tmp_path / "process-status.json").write_text(json.dumps(sidecar))
    outputs = render_calibration_reports([source], tmp_path / "cancel-status")
    document = outputs["html"].read_text()
    assert expected in document
    summary = json.loads(outputs["summary_json"].read_text())
    assert summary["sources"][0]["status"] == report_status
    assert summary["sources"][0]["process_status"] == sidecar
    assert summary["trials"][-1]["status"] == "cancelled"
    assert summary["candidates"][0]["completed_measured_trials"] == 2
    assert summary["candidates"][0]["measured_median_native_prefill_seconds"] == 15
    if report_status == "cancelled" and code == 130:
        assert f"class='process-status'>{expected}</span>" in document
    else:
        assert "Cancelled (exit 130)" not in document
