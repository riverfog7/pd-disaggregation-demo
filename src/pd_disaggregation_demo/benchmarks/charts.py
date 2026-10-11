import html
import json
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import BenchmarkReport, BenchmarkSample

COLORS = {"local": "#247BA0", "split": "#F18F01"}
PHASES = {
    "tokenize_ms": "Tokenization",
    "prefill_ms": "Prefill",
    "export_ms": "Save",
    "transfer_ms": "Transfer",
    "restore_ms": "Restore",
    "final_prompt_ms": "Final prompt token",
}


def latest_complete_pair(
    report: BenchmarkReport,
) -> tuple[BenchmarkSample, BenchmarkSample] | None:
    pairs = {}
    latest = None
    for sample in report.samples:
        if not sample.result.success:
            continue
        pair = pairs.setdefault((sample.prompt_tokens, sample.repeat), {})
        pair[sample.result.mode] = sample
        if "local" in pair and "split" in pair:
            latest = pair["local"], pair["split"]
    return latest


def _duration(value):
    return (
        value if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None
    )


def _ttft_overhead(metrics):
    ttft = _duration(metrics.ttft_ms)
    stages = [_duration(getattr(metrics, name)) for name in PHASES]
    if ttft is None or any(value is None for value in stages):
        return None
    return _duration(ttft - sum(stages))


def _after_first_token(metrics):
    ttft, total = _duration(metrics.ttft_ms), _duration(metrics.total_ms)
    return _duration(total - ttft) if ttft is not None and total is not None else None


def stage_comparison(report: BenchmarkReport) -> tuple[str, list[list], list[str]]:
    """Compare raw stage timers within one exact pair; never sum mode medians."""
    pair = latest_complete_pair(report)
    notes = [
        "Canonical corpus tokenization prepares identical token IDs once per prompt size, "
        "outside timed pairs. Per-request tokenization is normally 0 ms when IDs are supplied."
    ]
    rows = []
    if pair is None:
        caption = "Waiting for a complete local/split pair; unmatched requests are not combined."
        size = None
    else:
        local, split = pair
        size = local.prompt_tokens
        order = "local → split" if local.order < split.order else "split → local"
        caption = f"{size:,} prompt tokens · pair {local.repeat + 1} · {order}"
        if all(report.samples[-1] is not sample for sample in pair):
            caption += " · retaining this pair while its newer replacement is incomplete"
        metrics = local.result.metrics, split.result.metrics
        rows = [
            [label, *(_duration(getattr(value, name)) for value in metrics)]
            for name, label in PHASES.items()
        ]
        rows.extend(
            [
                ["Unattributed TTFT overhead", *(_ttft_overhead(value) for value in metrics)],
                [
                    "TTFT subtotal · through first token",
                    *(_duration(value.ttft_ms) for value in metrics),
                ],
                [
                    "After first token · client elapsed",
                    *(_after_first_token(value) for value in metrics),
                ],
                [
                    "Total request · client elapsed",
                    *(_duration(value.total_ms) for value in metrics),
                ],
                [
                    "Native decode · engine timing (not additive)",
                    *(_duration(value.decode_ms) for value in metrics),
                ],
            ]
        )
        notes.append(
            "TTFT is the measured time to the first token. After-first-token elapsed is "
            "total − TTFT and includes streaming and completion overhead. Native decode uses "
            "the engine's clock and overlaps this accounting; do not add it to the total."
        )
        if any(_ttft_overhead(value) is None for value in metrics):
            notes.append(
                "Some stage timers exceed measured TTFT or are unavailable. Their residual "
                "cannot be attributed; the timeline uses measured TTFT for that request "
                "instead of adding incompatible timers. Raw stage values remain above."
            )
    configurations = report.environment.get("configurations", [])
    preparation = next(
        (
            item
            for item in reversed(configurations)
            if size is None or item.get("prompt_tokens") == size
        ),
        None,
    )
    if preparation:
        observed = []
        for key, label in (
            ("corpus_tokenize_ms", "corpus tokenization"),
            ("startup_ms", "local configuration/startup"),
        ):
            value = _duration(preparation.get(key))
            if value is not None:
                observed.append(f"{label} {value:,.2f} ms")
        if observed:
            notes.append("Observed preparation outside timed pairs: " + "; ".join(observed) + ".")
    return caption, rows, notes


def stage_comparison_html(report: BenchmarkReport) -> str:
    caption, rows, notes = stage_comparison(report)
    body = "".join(
        '<tr style="border-bottom:1px solid #ddd">'
        f'<th scope="row" style="text-align:left;padding:7px">{html.escape(label)}</th>'
        + "".join(
            f'<td style="text-align:right;padding:7px;font-variant-numeric:tabular-nums">'
            f"{value:,.2f}</td>"
            if value is not None
            else '<td style="text-align:right;padding:7px">Unavailable</td>'
            for value in values
        )
        + "</tr>"
        for label, *values in rows
    )
    table = (
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
        '<thead><tr><th style="text-align:left;padding:7px">Stage</th>'
        '<th style="text-align:right;padding:7px;color:#247BA0">Local (ms)</th>'
        '<th style="text-align:right;padding:7px;color:#F18F01">Split (ms)</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
        if rows
        else ""
    )
    return (
        "<section><h3>Stage durations · latest complete pair</h3>"
        f"<p><strong>{html.escape(caption)}</strong></p>{table}"
        + "".join(f"<p>{html.escape(note)}</p>" for note in notes)
        + "</section>"
    )


def curves(reports: list[BenchmarkReport]):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Time to first token (ms)",
            "TTFT speedup (local / split), 95% paired CI",
            "Prefill tokens/s",
            "Decode tokens/s",
        ),
    )
    for report in reports:
        rows = report.summary
        prefix = report.config.label or report.run_id[:8]
        x = [r["prompt_tokens"] for r in rows]
        for mode in ("local", "split"):
            for metric, row, col in (
                ("ttft_ms", 1, 1),
                ("prefill_tps", 2, 1),
                ("decode_tps", 2, 2),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[r.get(f"{mode}_{metric}") for r in rows],
                        mode="lines+markers",
                        name=f"{prefix} · {mode}",
                        legendgroup=f"{prefix}-{mode}",
                        showlegend=(row, col) == (1, 1),
                        line={"color": COLORS[mode]},
                        connectgaps=False,
                    ),
                    row=row,
                    col=col,
                )
        speed = [r.get("speedup") for r in rows]
        upper = [None if r.get("ci_high") is None else r["ci_high"] - r["speedup"] for r in rows]
        lower = [None if r.get("ci_low") is None else r["speedup"] - r["ci_low"] for r in rows]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=speed,
                mode="lines+markers",
                name=f"{prefix} · paired speedup",
                error_y={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
                customdata=[r.get("pairs", 0) for r in rows],
                hovertemplate=(
                    "%{x} prompt tokens<br>%{y:.3f}×<br>%{customdata} complete pairs<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    figure.add_hline(y=1, line_dash="dash", line_color="#777", row=1, col=2)
    figure.update_xaxes(title_text="Prompt tokens", type="log")
    figure.update_layout(
        height=620, template="plotly_white", margin={"l": 55, "r": 25, "t": 60, "b": 45}
    )
    return figure


def phases(report: BenchmarkReport):
    figure = go.Figure()
    labels = [
        f"{s.prompt_tokens:,} · pair {s.repeat + 1} · {s.result.mode}" for s in report.samples
    ]
    overhead = [_ttft_overhead(sample.result.metrics) for sample in report.samples]
    for name, label in PHASES.items():
        figure.add_trace(
            go.Bar(
                y=labels,
                x=[
                    getattr(sample.result.metrics, name) if residual is not None else None
                    for sample, residual in zip(report.samples, overhead, strict=True)
                ],
                name=f"TTFT · {label}",
                orientation="h",
                hovertemplate=f"{label}: %{{x:.2f}} ms<extra></extra>",
            )
        )
    # Reported TTFT may also include request/network/scheduling overhead. Keep this visible.
    figure.add_trace(
        go.Bar(
            y=labels,
            x=overhead,
            name="TTFT · Unattributed overhead",
            orientation="h",
            hovertemplate="Unattributed TTFT overhead: %{x:.2f} ms<extra></extra>",
        )
    )
    if any(value is None for value in overhead):
        figure.add_trace(
            go.Bar(
                y=labels,
                x=[
                    _duration(sample.result.metrics.ttft_ms) if residual is None else None
                    for sample, residual in zip(report.samples, overhead, strict=True)
                ],
                name="Measured TTFT · incompatible stage timers",
                orientation="h",
            )
        )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=[_after_first_token(sample.result.metrics) for sample in report.samples],
            name="After first token · client elapsed",
            marker_color="#64748B",
            orientation="h",
            hovertemplate="After first token: %{x:.2f} ms<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            y=labels,
            x=[_duration(sample.result.metrics.ttft_ms) for sample in report.samples],
            name="First token / TTFT boundary",
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 18, "color": "#111"},
            hovertemplate="Measured TTFT: %{x:.2f} ms<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="stack",
        height=max(300, min(1500, 24 * len(report.samples) + 130)),
        template="plotly_white",
        title="Whole-request timeline · first token boundary separates TTFT and later elapsed time",
        xaxis_title="Elapsed time (ms)",
        yaxis={"autorange": "reversed"},
    )
    return figure


def transfer(report: BenchmarkReport):
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("State size (MiB)", "Transfer rate (Mbit/s)")
    )
    rows = report.summary
    x = [r["prompt_tokens"] for r in rows]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=[
                None if r.get("split_state_bytes") is None else r["split_state_bytes"] / 1024**2
                for r in rows
            ],
            mode="lines+markers",
            name="State MiB",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=[r.get("split_transfer_mbps") for r in rows],
            mode="lines+markers",
            name="Transfer Mbit/s",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(height=300, template="plotly_white")
    figure.update_xaxes(title_text="Prompt tokens", type="log")
    return figure


def memory_rows(report: BenchmarkReport) -> list[list]:
    return [
        [
            s.prompt_tokens,
            s.repeat + 1,
            s.result.mode,
            json.dumps(s.result.metrics.memory, ensure_ascii=False, default=str),
        ]
        for s in report.samples
    ]


def memory_chart(report: BenchmarkReport):
    figure = go.Figure()

    def flatten(value, prefix=""):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                yield from flatten(item, name)
            elif (
                name.endswith("bytes")
                and isinstance(item, (int, float))
                and any(
                    part in key for part in ("ram_", "gpu_", "rss_", "swap_", "metal_", "cgroup_")
                )
            ):
                yield name, item

    series = {}
    for index, sample in enumerate(report.samples):
        for name, value in flatten(sample.result.metrics.memory):
            series.setdefault(name, []).append((index + 1, value / 1024**3))
    for name, values in series.items():
        figure.add_trace(
            go.Scatter(
                x=[x for x, _ in values], y=[y for _, y in values], mode="lines+markers", name=name
            )
        )
    figure.update_layout(
        title="Memory snapshots and sampled peaks (short transients may be missed)",
        template="plotly_white",
        height=350,
        xaxis_title="Completed request",
        yaxis_title="GiB",
    )
    return figure
