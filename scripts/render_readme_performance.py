# /// script
# requires-python = ">=3.13"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Render the README's measured speedups from committed benchmark evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "benchmark-4b-q8-rerun-2026-10-12.json",
    "benchmark-4b-65k-2026-10-12.json",
)


def main():
    rows = []
    for filename in SOURCES:
        report = json.loads((ROOT / "reports" / filename).read_text())
        rows.extend(report["summary"])
    rows.sort(key=lambda row: row["prompt_tokens"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    figure, axes = plt.subplots(figsize=(11, 5.4), facecolor="white")
    figure.subplots_adjust(left=0.17, right=0.72, top=0.71, bottom=0.30)
    axes.axvspan(0, 1, color="#fff4ed", zorder=0)
    axes.axvspan(1, 3, color="#eef7fb", zorder=0)
    axes.axvline(1, color="#475569", linewidth=1.2, linestyle="--")
    for index, row in enumerate(rows):
        speedup, low, high = (row[key] for key in ("speedup", "ci_low", "ci_high"))
        color = "#136b91" if low > 1 else "#b84717"
        axes.errorbar(
            speedup,
            index,
            xerr=[[speedup - low], [high - speedup]],
            fmt="o",
            color=color,
            markersize=8,
            capsize=5,
            linewidth=2,
        )
        axes.text(
            1.04,
            index,
            f"{speedup:.2f}×  [{low:.2f}, {high:.2f}]",
            transform=axes.get_yaxis_transform(),
            va="center",
            color=color,
            fontsize=12,
            weight="bold",
        )
    axes.set_yticks(range(len(rows)), [f"{row['prompt_tokens']:,}" for row in rows])
    axes.set_ylim(len(rows) - 0.5, -0.5)
    axes.set_xlim(0, 3)
    axes.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    axes.set_ylabel("Prompt tokens", labelpad=14)
    axes.set_xlabel("Paired TTFT speedup · local / split", labelpad=12)
    axes.tick_params(axis="both", length=0, pad=8)
    axes.grid(axis="x", color="#cbd5e1", alpha=0.45)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.text(0.5, -0.78, "Split slower", ha="center", color="#b84717", fontsize=11)
    axes.text(2, -0.78, "Split faster", ha="center", color="#136b91", fontsize=11)
    axes.text(
        1.04,
        -0.78,
        "Speedup [95% paired CI]",
        transform=axes.get_yaxis_transform(),
        color="#475569",
        fontsize=11,
    )
    figure.text(
        0.06, 0.92, "Does remote prefill improve first-token latency?", size=18, weight="bold"
    )
    figure.text(
        0.06,
        0.85,
        "Qwen3.5-4B · M4 Pro + RTX 2080 · measured SSH route · 2026-10-12",
        size=11,
        color="#475569",
    )
    figure.text(
        0.06,
        0.075,
        "1k–8k: five pairs, microbatch 128. 65,535: four pairs, microbatch 2,048.\n"
        "Separate experiments with different settings; "
        "these points do not isolate a prompt-length crossover.",
        size=10,
        color="#475569",
        linespacing=1.7,
    )
    destination = ROOT / "docs" / "assets" / "performance-4b.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
