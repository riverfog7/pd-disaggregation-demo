import csv
import json

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
