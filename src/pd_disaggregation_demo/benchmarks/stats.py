import math
import random
import statistics

from ..models import GenerationResult
from .models import ONE_GBPS_NOTE, BenchmarkReport


def modeled_transfer_ms(result: GenerationResult) -> float | None:
    """Ideal payload time at 1,000 Mbit/s; unknown or invalid state size is unavailable."""
    size = result.metrics.state_bytes
    if result.mode != "split" or not result.success or size <= 0:
        return None
    return size / 125_000_000 * 1000


def modeled_split_ttft_ms(result: GenerationResult) -> float | None:
    metrics = result.metrics
    ideal = modeled_transfer_ms(result)
    if (
        ideal is None
        or metrics.ttft_ms is None
        or not math.isfinite(metrics.ttft_ms)
        or metrics.ttft_ms <= 0
        or not math.isfinite(metrics.transfer_ms)
        or not 0 <= metrics.transfer_ms <= metrics.ttft_ms
    ):
        return None
    return metrics.ttft_ms - metrics.transfer_ms + ideal


def paired_speedup(local: list[float], split: list[float], *, seed: int, draws: int) -> dict:
    """Geometric mean of within-pair TTFT ratios with a percentile bootstrap interval."""
    if (
        len(local) != len(split)
        or not local
        or any(not math.isfinite(v) or v <= 0 for v in local + split)
    ):
        raise ValueError(
            "Paired TTFT values must be finite, positive and have equal nonzero lengths"
        )
    logs = [math.log(a) - math.log(b) for a, b in zip(local, split, strict=True)]
    estimate = math.exp(statistics.mean(logs))
    if len(logs) < 2:
        return {
            "speedup": estimate,
            "ci_low": None,
            "ci_high": None,
            "supported_win": False,
            "pairs": len(logs),
        }
    rng = random.Random(seed)
    boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(draws))
    low, high = boot[int((draws - 1) * 0.025)], boot[int((draws - 1) * 0.975)]
    return {
        "speedup": estimate,
        "ci_low": low,
        "ci_high": high,
        "supported_win": low > 1,
        "pairs": len(logs),
    }


def _modeled_summary(report, samples):
    projections = []
    pairs = {}
    for sample in samples:
        if not sample.result.success:
            continue
        mode = sample.result.mode
        value = (
            modeled_split_ttft_ms(sample.result)
            if mode == "split"
            else sample.result.metrics.ttft_ms
        )
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        if mode == "split":
            projections.append(value)
        pairs.setdefault(sample.repeat, {})[mode] = value
    complete = [pair for pair in pairs.values() if "local" in pair and "split" in pair]
    row = {
        "modeled_1gbps_mbps": 1000,
        "modeled_1gbps_split_ttft_ms": statistics.median(projections) if projections else None,
        "modeled_1gbps_samples": len(projections),
        "modeled_1gbps_speedup": None,
        "modeled_1gbps_ci_low": None,
        "modeled_1gbps_ci_high": None,
        "modeled_1gbps_pairs": len(complete),
    }
    if complete:
        estimate = paired_speedup(
            [pair["local"] for pair in complete],
            [pair["split"] for pair in complete],
            seed=report.config.seed,
            draws=report.config.bootstrap_samples,
        )
        # A modeled interval must never set or replace the measured supported_win field.
        for key in ("speedup", "ci_low", "ci_high"):
            row[f"modeled_1gbps_{key}"] = estimate[key]
    return row


def add_modeled_projection(report: BenchmarkReport) -> BenchmarkReport:
    """Derive legacy report projections in memory, retaining every measured summary field."""
    for row in report.summary:
        if "modeled_1gbps_mbps" not in row:
            samples = [
                sample for sample in report.samples if sample.prompt_tokens == row["prompt_tokens"]
            ]
            row.update(_modeled_summary(report, samples))
    if ONE_GBPS_NOTE not in report.notes:
        report.notes.append(ONE_GBPS_NOTE)
    return report


def summarize(report: BenchmarkReport) -> list[dict]:
    rows = []
    for size in report.config.prompt_sizes:
        samples = [s for s in report.samples if s.prompt_tokens == size and s.result.success]
        pair_map = {}
        row = {"prompt_tokens": size}
        for mode in ("local", "split"):
            subset = [s for s in samples if s.result.mode == mode]
            row[f"{mode}_samples"] = len(subset)
            for name in (
                "ttft_ms",
                "prefill_tps",
                "decode_tps",
                "export_ms",
                "transfer_ms",
                "restore_ms",
                "state_bytes",
                "transfer_mbps",
                "total_ms",
            ):
                values = [getattr(s.result.metrics, name) for s in subset]
                values = [v for v in values if v is not None]
                row[f"{mode}_{name}"] = statistics.median(values) if values else None
            for sample in subset:
                if sample.result.metrics.ttft_ms and sample.result.metrics.ttft_ms > 0:
                    pair_map.setdefault(sample.repeat, {})[mode] = sample.result.metrics.ttft_ms
        pairs = [p for p in pair_map.values() if "local" in p and "split" in p]
        if pairs:
            row.update(
                paired_speedup(
                    [p["local"] for p in pairs],
                    [p["split"] for p in pairs],
                    seed=report.config.seed,
                    draws=report.config.bootstrap_samples,
                )
            )
        else:
            row.update(speedup=None, ci_low=None, ci_high=None, supported_win=False, pairs=0)
        row.update(_modeled_summary(report, samples))
        rows.append(row)
    return rows
