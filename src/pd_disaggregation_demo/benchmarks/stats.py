import math
import random
import statistics

from .models import BenchmarkReport


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
        rows.append(row)
    return rows
