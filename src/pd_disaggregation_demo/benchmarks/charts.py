import html
import json
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import ONE_GBPS_NOTE, BenchmarkReport, BenchmarkSample
from .stats import add_modeled_projection

COLORS = {"local": "#247BA0", "split": "#F18F01"}
PHASES = {
    "tokenize_ms": "Tokenization",
    "prefill_ms": "Prefill",
    "export_ms": "Save",
    "transfer_ms": "Transfer",
    "restore_ms": "Restore",
    "final_prompt_ms": "Final prompt token",
}


def _bottom_legend():
    # Fractional entry widths wrap into two columns as the Gradio panel resizes.
    return {
        "orientation": "h",
        "traceorder": "normal",
        "x": 0,
        "xanchor": "left",
        "y": -0.15,
        "yanchor": "top",
        "entrywidth": 0.5,
        "entrywidthmode": "fraction",
        "maxheight": 120,
        "font": {"size": 11},
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


def projection_html(report: BenchmarkReport) -> str:
    add_modeled_projection(report)
    rows = []
    for row in report.summary:
        value = row.get("modeled_1gbps_split_ttft_ms")
        latency = f"{value:,.2f}" if value is not None else "Unavailable"
        ratio = row.get("modeled_1gbps_speedup")
        estimate = f"{ratio:.3f}×" if ratio is not None else "Unavailable"
        low, high = row.get("modeled_1gbps_ci_low"), row.get("modeled_1gbps_ci_high")
        interval = (
            f"95% CI {low:.3f}–{high:.3f}×"
            if low is not None and high is not None
            else "95% CI unavailable"
        )
        rows.append(
            '<tr style="border-bottom:1px solid #ddd">'
            f'<td style="padding:7px">{row["prompt_tokens"]:,}</td>'
            f'<td style="padding:7px;text-align:right">{latency}</td>'
            f'<td style="padding:7px;text-align:right">{estimate}<br>{interval}'
            f"<br>{row.get('modeled_1gbps_pairs', 0)} complete pairs</td></tr>"
        )
    return (
        "<section><h3>Modeled ideal 1Gbps · optimistic projection</h3>"
        f"<p>{html.escape(ONE_GBPS_NOTE)}</p>"
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
        '<thead><tr><th style="text-align:left;padding:7px">Prompt tokens</th>'
        '<th style="text-align:right;padding:7px">Modeled split TTFT (ms)</th>'
        '<th style="text-align:right;padding:7px">Modeled local/split ratio</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def curves(reports: list[BenchmarkReport]):
    figure = make_subplots(
        rows=4,
        cols=1,
        vertical_spacing=0.10,
        subplot_titles=(
            "Time to first token (ms)",
            "TTFT speedup · 95% paired CI",
            "Prefill tokens/s",
            "Decode tokens/s",
        ),
    )
    for report in reports:
        add_modeled_projection(report)
        rows = report.summary
        full_prefix = report.config.label or report.run_id[:8]
        prefix = full_prefix[:16] + ("…" if len(full_prefix) > 16 else "")
        x = [r["prompt_tokens"] for r in rows]
        for mode in ("local", "split"):
            for metric, row, col in (
                ("ttft_ms", 1, 1),
                ("prefill_tps", 3, 1),
                ("decode_tps", 4, 1),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[r.get(f"{mode}_{metric}") for r in rows],
                        mode="lines+markers",
                        name=f"{prefix} · {mode}" if len(reports) > 1 else mode.title(),
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
                name=f"{prefix} · speedup" if len(reports) > 1 else "Paired speedup",
                error_y={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
                customdata=[r.get("pairs", 0) for r in rows],
                hovertemplate=(
                    "%{x} prompt tokens<br>%{y:.3f}×<br>%{customdata} complete pairs<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        if any(row.get("modeled_1gbps_split_ttft_ms") is not None for row in rows):
            modeled_name = f"{prefix} · 1Gbps model" if len(reports) > 1 else "1Gbps model"
            group = f"{report.run_id}-modeled-1gbps"
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[row.get("modeled_1gbps_split_ttft_ms") for row in rows],
                    mode="lines+markers",
                    name=modeled_name,
                    legendgroup=group,
                    line={"color": "#8B5CF6", "dash": "dash"},
                    connectgaps=False,
                    hovertemplate=(
                        "Modeled ideal 1Gbps<br>%{x} tokens<br>"
                        "Split TTFT %{y:.2f} ms<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            estimate = [row.get("modeled_1gbps_speedup") for row in rows]
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=estimate,
                    mode="lines+markers",
                    name=modeled_name,
                    legendgroup=group,
                    showlegend=False,
                    line={"color": "#8B5CF6", "dash": "dash"},
                    connectgaps=False,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            None
                            if row.get("modeled_1gbps_ci_high") is None
                            else row["modeled_1gbps_ci_high"] - value
                            for row, value in zip(rows, estimate, strict=True)
                        ],
                        "arrayminus": [
                            None
                            if row.get("modeled_1gbps_ci_low") is None
                            else value - row["modeled_1gbps_ci_low"]
                            for row, value in zip(rows, estimate, strict=True)
                        ],
                    },
                    customdata=[row.get("modeled_1gbps_pairs", 0) for row in rows],
                    hovertemplate=(
                        "Modeled ideal 1Gbps<br>%{x} tokens<br>"
                        "Local / projected split %{y:.3f}×<br>"
                        "%{customdata} complete pairs<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    figure.add_hline(y=1, line_dash="dash", line_color="#777", row=2, col=1)
    figure.update_xaxes(type="log", tickangle=0, nticks=4, automargin=True)
    figure.update_xaxes(title_text="Prompt tokens", row=4, col=1)
    figure.update_yaxes(automargin=True, nticks=4)
    figure.update_layout(
        height=1100,
        autosize=True,
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 45, "b": 200},
        legend={**_bottom_legend(), "y": -0.07},
    )
    figure.update_annotations(font_size=13)
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
                name=label,
                orientation="h",
                hovertemplate=f"{label}: %{{x:.2f}} ms<extra></extra>",
            )
        )
    # Reported TTFT may also include request/network/scheduling overhead. Keep this visible.
    figure.add_trace(
        go.Bar(
            y=labels,
            x=overhead,
            name="Other TTFT overhead",
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
                name="TTFT (unpartitioned)",
                orientation="h",
            )
        )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=[_after_first_token(sample.result.metrics) for sample in report.samples],
            name="After first token",
            marker_color="#64748B",
            orientation="h",
            hovertemplate="After first token: %{x:.2f} ms<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            y=labels,
            x=[_duration(sample.result.metrics.ttft_ms) for sample in report.samples],
            name="First token / TTFT",
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 18, "color": "#111"},
            hovertemplate="Measured TTFT: %{x:.2f} ms<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="stack",
        height=max(480, min(1500, 24 * len(report.samples) + 280)),
        autosize=True,
        template="plotly_white",
        title={"text": "Whole-request timeline", "font": {"size": 16}},
        margin={"l": 115, "r": 20, "t": 50, "b": 190},
        legend=_bottom_legend(),
        xaxis_title="Elapsed time (ms)",
        xaxis={"tickangle": 0, "nticks": 4, "automargin": True},
        yaxis={"autorange": "reversed", "automargin": True},
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
