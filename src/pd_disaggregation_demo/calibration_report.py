"""Render saved calibration evidence without running an engine or inferring causes."""

import argparse
import csv
import html
import io
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

GiB = 1024**3
MiB = 1024**2
NOTES = [
    "Completed measured trials alone contribute to comparison medians; warmup is separate.",
    "GPU utilization is sampled device activity. "
    "Low activity alone does not prove a disk bottleneck.",
    "Physical read counters, major faults, and memory/IO pressure provide additional evidence.",
    "Read rate is derived from consecutive same-phase counter samples, not from GPU utilization.",
    "Missing, dropped, and truncated telemetry remains explicit. Sampling can miss short peaks.",
    "CUDA prefill excludes the final canonical token; "
    "native counts and exercised chunks are shown.",
    "Times are native inference/export timers unless a column explicitly says client or wall.",
    "GPU activity phase summaries cover the full trial, including export and client overhead; "
    "they are not prefill-only kernel measurements.",
]


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _scaled(value: Any, divisor: float) -> float | None:
    value = _number(value)
    return value / divisor if value is not None else None


def _median(rows: list[dict], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return statistics.median(values) if values else None


def _placement(report: dict) -> str:
    placement = report.get("placement", {})
    profile = report.get("requested_profile", {})
    overrides = placement.get("tensor_overrides", [])
    output_overrides = []
    attention_overrides = []
    for pattern in overrides:
        expression, _, buffer = pattern.partition("=")
        try:
            if buffer == "CPU" and re.search(expression, "output.weight"):
                output_overrides.append(pattern)
            if buffer == "CPU" and all(
                re.search(expression, f"blk.0.{name}.weight")
                for name in ("attn_q", "attn_k", "attn_v", "attn_output")
            ):
                attention_overrides.append(pattern)
        except re.error:
            pass
    output_cpu = placement.get("cpu_output_projection") or bool(output_overrides)
    attention_cpu = placement.get("cpu_attention_projections") or bool(attention_overrides)
    suffix = ""
    projections = [
        name
        for name in ("attn_qkv", "attn_gate", "ssm_out")
        if any(name in pattern for pattern in overrides)
    ]
    if placement.get("cpu_recurrent_projections") or len(projections) == 3:
        suffix = "+recurrent"
    elif projections:
        names = {"attn_qkv": "QKV", "attn_gate": "gate", "ssm_out": "SSM-out"}
        suffix = "+" + "/".join(names[name] for name in projections)
    elif remaining := [
        pattern
        for pattern in overrides
        if pattern not in output_overrides and pattern not in attention_overrides
    ]:
        suffix = "+" + "; ".join(remaining)
    if attention_cpu:
        suffix += "+attention"
    if output_cpu:
        suffix += "+output"
    count = placement.get("cpu_ffn", profile.get("cpu_ffn", "?"))
    return f"{str(report.get('backend', 'unknown')).upper()}: FFN{count}{suffix}"


def _timeline(candidate: dict, source_path: Path) -> dict:
    metadata = candidate.get("gpu_disk_telemetry", {})
    result = {
        "metadata": metadata,
        "samples": [],
        "available": False,
        "reason": "No GPU/disk timeline recorded",
    }
    raw_path = metadata.get("jsonl_path")
    if not raw_path:
        return result
    raw = Path(raw_path)
    choices = [
        raw if raw.is_absolute() else source_path.parent / raw,
        source_path.parent / raw.parent.name / raw.name,
        source_path.parent / f"{source_path.stem}.telemetry" / raw.name,
        source_path.parent / raw.name,
    ]
    path = next((choice for choice in choices if choice.is_file()), None)
    if path is None:
        result["reason"] = f"Timeline file unavailable: {raw_path}"
        return result
    samples: list[dict] = []
    with path.open() as handle:
        header = json.loads(next(handle))
        columns = header.get("columns")
        if not isinstance(columns, list) or "elapsed_ms" not in columns or "phase" not in columns:
            raise ValueError(f"Malformed timeline header: {path}")
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(f"Malformed timeline row: {path}")
            sample = dict(zip(columns, row, strict=True))
            previous = samples[-1] if samples else None
            rate = None
            if previous and previous.get("phase") == sample.get("phase"):
                now, before = _number(sample.get("elapsed_ms")), _number(previous.get("elapsed_ms"))
                read, old_read = (
                    _number(sample.get("native_read_bytes")),
                    _number(previous.get("native_read_bytes")),
                )
                if (
                    now is not None
                    and before is not None
                    and now > before
                    and read is not None
                    and old_read is not None
                    and read >= old_read
                ):
                    rate = (read - old_read) / MiB / ((now - before) / 1000)
            sample["physical_read_mib_per_second"] = rate
            samples.append(sample)
    result.update(
        samples=samples,
        available=bool(samples),
        reason=None,
        resolved_jsonl_path=str(path),
        columns=columns,
    )
    return result


def _trial_row(source_id: str, label: str, candidate: dict, trial: dict, timeline: dict) -> dict:
    manifest = trial.get("manifest", {})
    result = trial.get("result", {})
    metrics = manifest.get("timings") or result.get("metrics", {})
    peak = trial.get("peak_telemetry", {})
    after = trial.get("telemetry_after", {})
    delta = trial.get("io_delta", {})
    phase_name = f"{trial.get('kind')}:{trial.get('repeat')}"
    phase = trial.get("gpu_disk_summary") or timeline.get("metadata", {}).get("phases", {}).get(
        phase_name, {}
    )
    pressure = trial.get("pressure_stall_delta_us", {})
    gpu_total = _number(after.get("gpu_total_bytes"))
    free = _number(peak.get("gpu_free_min_bytes"))
    prompt_n = trial.get("native_prompt_n", metrics.get("prompt_n"))
    ubatch = candidate.get("ubatch_size")
    chunk = trial.get("largest_exercised_prompt_chunk")
    if chunk is None and isinstance(prompt_n, int) and isinstance(ubatch, int):
        chunk = min(prompt_n, ubatch)
    return {
        "source_id": source_id,
        "placement": label,
        "ubatch_size": ubatch,
        "kind": trial.get("kind"),
        "repeat": trial.get("repeat"),
        "status": trial.get("status"),
        "native_prompt_tokens": prompt_n,
        "largest_exercised_prompt_chunk": chunk,
        "native_cache_tokens": trial.get("native_cache_n", metrics.get("cache_n")),
        "native_saved_tokens": trial.get("native_saved_n"),
        "native_prefill_seconds": _scaled(
            metrics.get("native_prefill_ms", metrics.get("prefill_ms")), 1000
        ),
        "native_prefill_tps": _number(metrics.get("prefill_tps")),
        "client_prefill_seconds": _scaled(manifest.get("timings", {}).get("prefill_ms"), 1000),
        "native_export_seconds": _scaled(metrics.get("native_save_ms"), 1000),
        "client_export_seconds": _scaled(metrics.get("export_ms"), 1000),
        "native_decode_seconds": _scaled(metrics.get("decode_ms"), 1000),
        "trial_wall_seconds": _scaled(trial.get("wall_ms"), 1000),
        "operation_wall_seconds": _scaled(trial.get("operation_wall_ms"), 1000),
        "gpu_free_min_gib": _scaled(free, GiB),
        "device_vram_used_peak_gib": (gpu_total - free) / GiB
        if gpu_total is not None and free is not None
        else None,
        "host_available_min_gib": _scaled(peak.get("ram_available_min_bytes"), GiB),
        "cgroup_anon_peak_gib": _scaled(peak.get("cgroup_anon_bytes_max"), GiB),
        "cgroup_file_peak_gib": _scaled(peak.get("cgroup_file_bytes_max"), GiB),
        "native_rss_peak_gib": _scaled(peak.get("native_rss_bytes_max"), GiB),
        "physical_read_gib": _scaled(delta.get("native_read_bytes"), GiB),
        "major_faults": _number(delta.get("native_major_faults")),
        "gpu_activity_time_weighted_percent": _number(
            phase.get("time_weighted_gpu_utilization_percent")
        ),
        "gpu_activity_below_20_fraction": _number(
            phase.get("fraction_gpu_utilization_below_20_percent")
        ),
        "gpu_timeline_coverage_seconds": _scaled(phase.get("gpu_coverage_ms"), 1000),
        "telemetry_gap_max_seconds": _scaled(phase.get("gap_max_ms"), 1000),
        "memory_some_stall_seconds": _scaled(pressure.get("memory_some_total_us"), 1e6),
        "memory_full_stall_seconds": _scaled(pressure.get("memory_full_total_us"), 1e6),
        "io_some_stall_seconds": _scaled(pressure.get("io_some_total_us"), 1e6),
        "io_full_stall_seconds": _scaled(pressure.get("io_full_total_us"), 1e6),
        "error": trial.get("error") or trial.get("final_sample_error"),
    }


def summarize_calibrations(inputs: Sequence[Path]) -> dict:
    sources, candidates, trials, timelines = [], [], [], []
    for index, path in enumerate(inputs):
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not isinstance(report.get("candidates"), list):
            raise ValueError(f"Not a saved calibration report: {path}")
        source_id, label = f"source-{index + 1}", _placement(report)
        run_label = f"Run {index + 1}"
        process_path = path.parent / "process-status.json"
        process_status = json.loads(process_path.read_text()) if process_path.is_file() else None
        if process_status is not None and not isinstance(process_status, dict):
            raise ValueError(f"Malformed process status sidecar: {process_path}")
        sources.append(
            {
                "source_id": source_id,
                "run_label": run_label,
                "path": str(path.resolve()),
                "status": report.get("status"),
                "error": report.get("error"),
                "backend": report.get("backend"),
                "placement": report.get("placement", {}),
                "requested_profile": report.get("requested_profile", {}),
                "environment": report.get("environment", {}),
                "process_status": process_status,
                "process_status_path": str(process_path.resolve())
                if process_status is not None
                else None,
                "metadata": {
                    key: value
                    for key, value in report.items()
                    if any(
                        word in key.lower() for word in ("image", "revision", "sha", "synthetic")
                    )
                },
                "prompt_tokens": report.get("prompt_tokens"),
                "started_at": report.get("started_at"),
            }
        )
        for candidate in report["candidates"]:
            timeline = _timeline(candidate, path)
            timelines.append(
                {
                    "source_id": source_id,
                    "placement": label,
                    "ubatch_size": candidate.get("ubatch_size"),
                    **timeline,
                }
            )
            rows = [
                _trial_row(source_id, label, candidate, trial, timeline)
                for trial in candidate.get("trials", [])
            ]
            trials.extend(rows)
            measured = [
                row for row in rows if row["kind"] == "measured" and row["status"] == "completed"
            ]
            estimate = candidate.get("estimate", {})
            devices = estimate.get("devices", {})
            totals = [
                sum(item.get(key, 0) for key in ("model_bytes", "context_bytes", "compute_bytes"))
                for item in devices.values()
            ]
            summary = {
                "source_id": source_id,
                "run_label": run_label,
                "placement": label,
                "prompt_tokens": report.get("prompt_tokens"),
                "model_id": candidate.get("requested_profile", {}).get(
                    "model_id", report.get("requested_profile", {}).get("model_id")
                ),
                "cache_type": candidate.get("requested_profile", {}).get(
                    "cache_type", report.get("requested_profile", {}).get("cache_type")
                ),
                "context_size": candidate.get("requested_profile", {}).get(
                    "context_size", report.get("requested_profile", {}).get("context_size")
                ),
                "prompt_sha256": candidate.get("prompt_sha256"),
                "cgroup_limit_bytes": (
                    candidate.get("cgroup_before")
                    or candidate.get("preflight", {}).get("cgroup")
                    or report.get("environment", {}).get("cgroup_before")
                    or {}
                ).get("limit_bytes"),
                "ubatch_size": candidate.get("ubatch_size"),
                "status": candidate.get("status"),
                "completed_measured_trials": len(measured),
                "startup_seconds": _scaled(candidate.get("startup_ms"), 1000),
                "estimated_device_total_gib": sum(totals) / GiB if totals else None,
                "timeline_available": timeline["available"],
                "timeline_dropped_samples": timeline["metadata"].get("dropped_samples"),
                "timeline_truncated": timeline["metadata"].get("truncated"),
                "timeline_unavailable_reason": timeline["reason"],
                "effective_settings": candidate.get("effective_settings"),
                "error": candidate.get("error"),
                "skip_reasons": candidate.get("preflight", {}).get("reasons", []),
            }
            summary.update(
                {
                    key: candidate.get("requested_profile", {}).get(
                        key, report.get("requested_profile", {}).get(key)
                    )
                    for key in ("threads", "gpu_layers", "batch_size")
                }
            )
            for key in rows[0] if rows else []:
                if key not in {
                    "source_id",
                    "placement",
                    "ubatch_size",
                    "kind",
                    "repeat",
                    "status",
                    "error",
                }:
                    summary[f"measured_median_{key}"] = _median(measured, key)
            candidates.append(summary)
    return {
        "schema_version": 1,
        "notes": NOTES,
        "sources": sources,
        "candidates": candidates,
        "trials": trials,
        "timelines": timelines,
    }


def _csv(rows: list[dict]) -> str:
    output = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def value(item):
        if item is None:
            return "unavailable"
        if isinstance(item, float):
            return f"{item:.3g}"
        return html.escape(str(item))

    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{value(row.get(key))}</td>" for key, _ in columns) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _html(summary: dict) -> str:
    metrics = (
        ("native_prefill_tps", "Prefill throughput", "Tokens/s"),
        ("native_prefill_seconds", "Prefill duration", "Seconds"),
        ("gpu_free_min_gib", "GPU free memory minimum", "GiB"),
        ("host_available_min_gib", "Host available memory minimum", "GiB"),
        ("physical_read_gib", "Physical disk reads", "GiB"),
        ("major_faults", "Major page faults", "Faults"),
    )
    ticks = sorted({int(row["ubatch_size"]) for row in summary["candidates"]})
    sources = {source["source_id"]: source for source in summary["sources"]}
    caps = {
        candidate.get("cgroup_limit_bytes")
        for candidate in summary["candidates"]
        if candidate.get("cgroup_limit_bytes") is not None
    }
    groups: dict[tuple, list[dict]] = {}
    for candidate in summary["candidates"]:
        if not candidate["completed_measured_trials"]:
            continue
        source = sources[candidate["source_id"]]
        profile = tuple(
            candidate.get(key)
            for key in (
                "model_id",
                "cache_type",
                "prompt_tokens",
                "context_size",
                "placement",
                "threads",
                "gpu_layers",
                "batch_size",
            )
        )
        environment = source.get("environment", {})
        key = (
            *profile,
            environment.get("engine_revision"),
            environment.get("model_sha256"),
            candidate.get("cgroup_limit_bytes"),
            candidate.get("prompt_sha256"),
            json.dumps(source.get("placement", {}).get("tensor_overrides", [])),
        )
        if any(value is None for value in profile):
            key = (*key, source["source_id"])
        groups.setdefault(key, []).append(candidate)
    plots = []
    for metric, title, unit in metrics:
        fig = go.Figure()
        for group in groups.values():
            rows = sorted(group, key=lambda row: row["ubatch_size"])
            runs = list(dict.fromkeys(row["run_label"] for row in rows))
            label = " + ".join(runs) if len(summary["sources"]) > 1 else rows[0]["placement"]
            cap = rows[0].get("cgroup_limit_bytes")
            if len(caps) > 1 and cap is not None:
                label += f" · {cap / GiB:g} GiB RAM cap"
            fig.add_trace(
                go.Scatter(
                    x=[row["ubatch_size"] for row in rows],
                    y=[row.get(f"measured_median_{metric}") for row in rows],
                    mode="lines+markers",
                    name=html.escape(label),
                    customdata=[[row["run_label"], row["prompt_tokens"]] for row in rows],
                    hovertemplate="Microbatch %{x:.0f}<br>%{y:.3g} "
                    + unit
                    + "<br>%{customdata[0]} · %{customdata[1]} prompt tokens"
                    + "<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_xaxes(
            title_text="Configured microbatch",
            type="log",
            tickmode="array",
            tickvals=ticks,
            ticktext=[str(value) for value in ticks],
        )
        fig.update_yaxes(title_text=unit)
        fig.update_layout(
            autosize=True,
            height=340,
            template="plotly_white",
            margin={"l": 62, "r": 12, "t": 12, "b": 95},
            font={"size": 12},
            legend={"orientation": "h", "y": -0.34},
        )
        chart = fig.to_html(
            full_html=False,
            include_plotlyjs=not plots,
            config={"responsive": True, "displayModeBar": False},
        )
        plots.append(f"<section class='chart-card'><h3>{html.escape(title)}</h3>{chart}</section>")
    comparison = "<div class='chart-grid'>" + "".join(plots) + "</div>"
    timelines = []
    for timeline in summary["timelines"]:
        if not timeline["available"]:
            continue
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Sampled GPU activity (%)", "Observed physical disk read rate (MiB/s)"),
        )
        samples = timeline["samples"]
        phases = list(dict.fromkeys(sample.get("phase") for sample in samples))
        for phase in phases:
            points = [sample for sample in samples if sample.get("phase") == phase]
            for row, key in ((1, "gpu_utilization_percent"), (2, "physical_read_mib_per_second")):
                figure.add_trace(
                    go.Scatter(
                        x=[_scaled(p.get("elapsed_ms"), 1000) for p in points],
                        y=[p.get(key) for p in points],
                        mode="lines",
                        name=str(phase),
                        legendgroup=str(phase),
                        showlegend=row == 1,
                        connectgaps=False,
                    ),
                    row=row,
                    col=1,
                )
        figure.update_yaxes(range=[0, 100], row=1, col=1)
        figure.update_xaxes(title_text="Seconds since candidate sampler started", row=2, col=1)
        figure.update_layout(
            height=580,
            template="plotly_white",
            autosize=True,
            margin={"l": 62, "r": 16, "t": 42, "b": 110},
            font={"size": 12},
            legend={"orientation": "h", "y": -0.22},
        )
        source = sources[timeline["source_id"]]
        prompt = source.get("prompt_tokens")
        count = f"{prompt:,}" if isinstance(prompt, int) else "unavailable"
        title = (
            f"{source['run_label']} · {html.escape(timeline['placement'])} · "
            f"{count} prompt tokens · microbatch {timeline['ubatch_size']}"
        )
        chart = figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displayModeBar": False},
        )
        timelines.append(f"<section class='timeline-card'><h3>{title}</h3>{chart}</section>")
    measured_rows = [row for row in summary["candidates"] if row["completed_measured_trials"]]
    measured = _table(
        measured_rows,
        [
            ("run_label", "Run"),
            ("prompt_tokens", "Prompt tokens"),
            ("placement", "CPU placement"),
            ("ubatch_size", "Microbatch"),
            ("measured_median_native_prefill_tps", "Tokens/s"),
            ("measured_median_native_prefill_seconds", "Prefill s"),
            ("measured_median_gpu_activity_time_weighted_percent", "GPU % (whole trial)"),
        ],
    )
    if not measured_rows:
        measured = "<p>No completed measured trials in these reports.</p>"
    candidates = _table(
        summary["candidates"],
        [
            ("placement", "Placement"),
            ("ubatch_size", "Configured microbatch"),
            ("status", "Evidence status"),
            ("completed_measured_trials", "Measured trials"),
            ("startup_seconds", "Startup s"),
            ("measured_median_native_prefill_seconds", "Native prefill s"),
            ("measured_median_native_prefill_tps", "Native tokens/s"),
            ("measured_median_gpu_activity_time_weighted_percent", "Mean sampled GPU %"),
            ("measured_median_gpu_activity_below_20_fraction", "Fraction GPU <20%"),
            ("skip_reasons", "Skip reasons"),
            ("error", "Failure"),
        ],
    )
    trials = _table(
        summary["trials"],
        [
            ("placement", "Placement"),
            ("ubatch_size", "Microbatch"),
            ("kind", "Phase"),
            ("repeat", "Repeat index"),
            ("status", "Status"),
            ("native_prompt_tokens", "Native prompt tokens"),
            ("largest_exercised_prompt_chunk", "Largest exercised chunk"),
            ("native_prefill_seconds", "Native prefill s"),
            ("native_prefill_tps", "Native tokens/s"),
            ("native_export_seconds", "Native export s"),
            ("client_export_seconds", "Client export s"),
            ("native_decode_seconds", "Native decode s"),
            ("physical_read_gib", "Physical reads GiB"),
            ("major_faults", "Major faults"),
            ("memory_full_stall_seconds", "Full memory stall s"),
            ("io_full_stall_seconds", "Full IO stall s"),
            ("device_vram_used_peak_gib", "Device VRAM peak GiB"),
            ("gpu_free_min_gib", "GPU free min GiB"),
            ("host_available_min_gib", "Host available min GiB"),
            ("cgroup_anon_peak_gib", "Cgroup anon peak GiB"),
            ("cgroup_file_peak_gib", "Cgroup file peak GiB"),
        ],
    )
    evidence = html.escape(
        json.dumps(
            {
                "sources": summary["sources"],
                "timelines": [
                    {key: value for key, value in item.items() if key != "samples"}
                    for item in summary["timelines"]
                ],
            },
            indent=2,
        )
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in NOTES)
    contexts = []
    for source in summary["sources"]:
        profile = source.get("requested_profile", {})
        model = html.escape(str(profile.get("model_id", "model unavailable")))
        cache = html.escape(str(profile.get("cache_type", "unavailable")))
        prompt = source.get("prompt_tokens")
        count = f"{prompt:,}" if isinstance(prompt, int) else "unavailable"
        source_caps = sorted(
            {
                candidate["cgroup_limit_bytes"]
                for candidate in summary["candidates"]
                if candidate["source_id"] == source["source_id"]
                and candidate.get("cgroup_limit_bytes") is not None
            }
        )
        cap_text = " / ".join(f"{cap / GiB:g}" for cap in source_caps)
        cap_label = f" · {cap_text} GiB RAM cap" if cap_text else ""
        code = (source.get("process_status") or {}).get("container_exit_code")
        process_class = "process-status"
        if not isinstance(code, int) or isinstance(code, bool):
            process_label = "Process exit status unknown"
        elif code != 0:
            completed = source.get("status") == "completed" and any(
                candidate["source_id"] == source["source_id"]
                and candidate["completed_measured_trials"]
                for candidate in summary["candidates"]
            )
            process_label = (
                f"Trials recorded; process cleanup failed (exit {code})"
                if completed
                else f"Process failed (exit {code})"
            )
            process_class += " process-failure"
        else:
            process_label = "Process exited cleanly (exit 0)"
        contexts.append(
            f"<p class='run-context'><strong>{source['run_label']}</strong> · {model} · "
            f"{count} prompt tokens · {cache} cache{cap_label}"
            f"<br><span class='{process_class}'>{process_label}</span></p>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Microbatch calibration evidence</title><style>"
        "body{font:15px system-ui;margin:0;padding:24px;color:#172435}"
        "main{max-width:1440px;margin:auto}h1{font-size:25px;margin:0 0 12px}"
        "h2{font-size:20px;margin:26px 0 10px}h3{font-size:16px;margin:12px 14px}"
        ".chart-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}"
        ".chart-card,.timeline-card{min-width:0;border:1px solid #dce2e8;border-radius:8px}"
        ".timeline-card{margin:16px 0}.plotly-graph-div{max-width:100%}"
        ".scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;margin:20px 0}"
        "td,th{padding:9px;border:1px solid #dce2e8;text-align:left;white-space:nowrap}"
        "th{background:#edf2f7;white-space:normal}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "details{margin:20px 0}summary{cursor:pointer;font-weight:600}"
        ".run-context{margin:6px 0;overflow-wrap:anywhere}"
        ".process-status{font-size:13px;color:#536274}.process-failure{color:#a12a2a}"
        "@media(max-width:1100px){.chart-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}"
        "@media(max-width:720px){body{padding:16px}.chart-grid{grid-template-columns:minmax(0,1fr)}}"
        "</style></head><body><main><h1>Microbatch calibration results</h1>"
        "<p>Completed measured trials · median values. Placement labels name CPU weights.</p>"
        + "".join(contexts)
        + measured
        + comparison
        + (
            "<h2>GPU activity and disk reads · whole trial</h2>" + "".join(timelines)
            if timelines
            else ""
        )
        + "<details><summary>All candidates, estimated skips, and failures</summary>"
        + candidates
        + "</details><details><summary>Startup, warmup, measured inference, and export</summary>"
        + trials
        + "</details><details><summary>Method and interpretation</summary><ul>"
        + notes
        + "</ul></details><details><summary>Source metadata and telemetry coverage</summary><pre>"
        + evidence
        + "</pre></details></main></body></html>"
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_calibration_reports(inputs: Sequence[Path], output_prefix: Path) -> dict[str, Path]:
    if not inputs:
        raise ValueError("At least one saved calibration report is required")
    summary = summarize_calibrations(inputs)
    outputs = {
        "summary_json": Path(f"{output_prefix}.summary.json"),
        "summary_csv": Path(f"{output_prefix}.summary.csv"),
        "trials_csv": Path(f"{output_prefix}.trials.csv"),
        "html": Path(f"{output_prefix}.html"),
    }
    _atomic_write(outputs["summary_json"], json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _atomic_write(outputs["summary_csv"], _csv(summary["candidates"]))
    _atomic_write(outputs["trials_csv"], _csv(summary["trials"]))
    _atomic_write(outputs["html"], _html(summary))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    options = parser.parse_args()
    for name, path in render_calibration_reports(options.inputs, options.output_prefix).items():
        print(f"{name}: {path}")
