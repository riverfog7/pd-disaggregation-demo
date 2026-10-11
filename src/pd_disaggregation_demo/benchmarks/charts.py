import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import BenchmarkReport

COLORS = {"local": "#247BA0", "split": "#F18F01"}
PHASES = {
    "tokenize_ms": "Tokenize",
    "prefill_ms": "Prefill",
    "export_ms": "Save",
    "transfer_ms": "Transfer",
    "restore_ms": "Restore",
    "final_prompt_ms": "Final token",
}


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
    for name, label in PHASES.items():
        figure.add_trace(
            go.Bar(
                y=[
                    f"{s.prompt_tokens:,} · pair {s.repeat + 1} · {s.result.mode}"
                    for s in report.samples
                ],
                x=[getattr(s.result.metrics, name) for s in report.samples],
                name=label,
                orientation="h",
            )
        )
    # Reported TTFT may also include request/network/scheduling overhead. Keep this visible.
    figure.add_trace(
        go.Bar(
            y=[
                f"{s.prompt_tokens:,} · pair {s.repeat + 1} · {s.result.mode}"
                for s in report.samples
            ],
            x=[
                max(
                    0,
                    (s.result.metrics.ttft_ms or 0)
                    - sum(getattr(s.result.metrics, name) for name in PHASES),
                )
                for s in report.samples
            ],
            name="Other TTFT overhead",
            orientation="h",
        )
    )
    figure.update_layout(
        barmode="stack",
        height=max(300, min(1500, 24 * len(report.samples) + 130)),
        template="plotly_white",
        title="First-token phase timeline (measured ms)",
        xaxis_title="Time (ms)",
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
