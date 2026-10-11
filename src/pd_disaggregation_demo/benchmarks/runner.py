import asyncio
import hashlib
import platform
import sys
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from ..models import GenerationRequest, Mode
from .models import BenchmarkConfig, BenchmarkEvent, BenchmarkReport, BenchmarkSample, SkippedPoint
from .stats import summarize
from .storage import save_report

CORPUS = (
    "A distributed inference experiment compares two machines using the same sequence of tokens. "
    "The prefill stage processes an existing document, and decoding produces new tokens in order. "
    "We measure elapsed time, memory, transfer size, and throughput to explain the result. "
    "Each trial starts with a fresh prompt cache and uses a deterministic sampling configuration.\n"
)


async def exact_tokens(pipeline, size: int) -> list[int]:
    """Tokenize a fixed text corpus and truncate canonical IDs to the exact requested size."""
    copies = max(1, size // 50)
    for _ in range(16):
        tokens = await pipeline.tokenize(CORPUS * copies)
        if len(tokens) >= size:
            return list(tokens[:size])
        copies *= 2
    raise ValueError("Tokenizer did not return enough tokens for the deterministic corpus")


async def _request(pipeline, ids, mode, output, seed):
    async for event in pipeline.generate(
        GenerationRequest(
            token_ids=ids,
            mode=mode,
            max_tokens=output,
            temperature=0,
            seed=seed,
            ignore_eos=True,
        )
    ):
        yield event


def _check_result(result, size, output):
    if not result.success:
        raise RuntimeError(result.error or "Generation failed")
    if result.prompt_tokens != size or result.generated_tokens != output:
        raise RuntimeError(
            f"Token budget mismatch: prompt={result.prompt_tokens}/{size}, "
            f"output={result.generated_tokens}/{output}"
        )
    if result.metrics.ttft_ms is None or result.metrics.ttft_ms <= 0:
        raise RuntimeError("Runtime returned no positive measured TTFT")
    if result.mode == "split" and (
        result.metrics.cache_n != size - 1 or result.metrics.prompt_n != 1
    ):
        raise RuntimeError("Split boundary mismatch: expected restored N-1 and final prompt_n=1")


def _is_admission(exc):
    message = str(exc).lower()
    return type(exc).__name__ == "AdmissionError" or any(
        phrase in message
        for phrase in (
            "insufficient memory",
            "memory reserve",
            "unsafe context",
            "not enough memory",
            "memory budget",
        )
    )


async def run_benchmark(
    pipeline, config: BenchmarkConfig, settings, *, cancel_event: asyncio.Event | None = None
) -> AsyncIterator[BenchmarkEvent]:
    """Stream progress and persist a complete or partial report, without a local fallback.

    Native runtime TTFT is authoritative; timings are never reconstructed from stage estimates.
    Cancellation via an event is checked before requests, and Pipeline.cancel interrupts active IO.
    """
    report = BenchmarkReport(
        config=config,
        environment={
            "platform": platform.platform(),
            "python": sys.version,
            "engine_revision": settings.engine_revision,
            "git_revision": settings.git_revision,
            "remote_url": settings.remote_url,
            "configurations": [],
            "prompt_sha256": {},
        },
    )

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    active_request = {}
    if not config.warmup:
        report.notes[0] = (
            "Warmup disabled; first measured requests may include cold-weight overhead."
        )
    try:
        health = await pipeline.health()
        report.environment["health_before"] = health
        yield BenchmarkEvent(
            kind="progress", stage="prepare", message="Preparing paired trials", report=report
        )
        for size in config.prompt_sizes:
            if cancelled():
                raise asyncio.CancelledError()
            context = max(config.profile.context_size, size + config.output_tokens)
            if context > 262144:
                reason = "Prompt plus output exceeds the maximum supported context (262144)"
                report.skipped.append(SkippedPoint(prompt_tokens=size, reason=reason))
                yield BenchmarkEvent(
                    kind="skipped", prompt_tokens=size, message=reason, report=report
                )
                continue
            profile = config.profile.model_copy(update={"context_size": context})
            yield BenchmarkEvent(
                kind="progress",
                stage="configure",
                prompt_tokens=size,
                message=f"Admitting context {context:,} and loading weights",
                report=report,
            )
            start = time.perf_counter()
            try:
                await pipeline.configure(profile)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_admission(exc):
                    raise
                reason = f"Configuration/admission rejected: {exc}"
                report.skipped.append(SkippedPoint(prompt_tokens=size, reason=reason))
                yield BenchmarkEvent(
                    kind="skipped", prompt_tokens=size, message=reason, report=report
                )
                continue
            report.environment["configurations"].append(
                {
                    "prompt_tokens": size,
                    "profile": profile.model_dump(),
                    "startup_ms": (time.perf_counter() - start) * 1000,
                }
            )
            token_start = time.perf_counter()
            ids = await exact_tokens(pipeline, size)
            report.environment["prompt_sha256"][str(size)] = hashlib.sha256(
                ",".join(map(str, ids)).encode()
            ).hexdigest()
            report.environment["configurations"][-1]["corpus_tokenize_ms"] = (
                time.perf_counter() - token_start
            ) * 1000
            if config.warmup:
                try:
                    for mode in ("local", "split"):
                        active_request = {"prompt_tokens": size, "stage": "warmup", "mode": mode}
                        if cancelled():
                            raise asyncio.CancelledError()
                        yield BenchmarkEvent(
                            kind="progress",
                            stage="warmup",
                            mode=mode,
                            prompt_tokens=size,
                            message=f"Warming {mode} weights separately",
                            report=report,
                        )
                        result = None
                        async for event in _request(pipeline, ids[:32], mode, 1, config.seed):
                            if cancelled():
                                raise asyncio.CancelledError()
                            if event.kind == "error":
                                raise RuntimeError(event.text or str(event.data))
                            if event.kind == "result":
                                result = event.result
                        if result is None or not result.success:
                            raise RuntimeError(result.error if result else f"{mode} warmup failed")
                        report.warmups.append(result)
                except Exception as exc:
                    if not _is_admission(exc):
                        raise
                    reason = f"Remote/local admission rejected: {exc}"
                    report.skipped.append(SkippedPoint(prompt_tokens=size, reason=reason))
                    yield BenchmarkEvent(
                        kind="skipped", prompt_tokens=size, message=reason, report=report
                    )
                    continue
            for repeat in range(config.repeats):
                modes: tuple[Mode, Mode] = (
                    ("local", "split") if repeat % 2 == 0 else ("split", "local")
                )
                for order, mode in enumerate(modes):
                    active_request = {
                        "prompt_tokens": size,
                        "repeat": repeat,
                        "mode": mode,
                        "stage": "measured",
                    }
                    if cancelled():
                        raise asyncio.CancelledError()
                    yield BenchmarkEvent(
                        kind="progress",
                        stage="request",
                        mode=mode,
                        repeat=repeat,
                        prompt_tokens=size,
                        message=f"{size:,} tokens · pair {repeat + 1}/{config.repeats} · {mode}",
                        report=report,
                    )
                    result = None
                    async for event in _request(
                        pipeline, ids, mode, config.output_tokens, config.seed
                    ):
                        if cancelled():
                            raise asyncio.CancelledError()
                        if event.kind == "stage":
                            yield BenchmarkEvent(
                                kind="stage",
                                stage=event.stage,
                                mode=mode,
                                repeat=repeat,
                                prompt_tokens=size,
                                data=event.data,
                                message=event.stage,
                                report=report,
                            )
                        elif event.kind == "error":
                            raise RuntimeError(event.text or str(event.data))
                        elif event.kind == "result":
                            result = event.result
                    if result is None:
                        raise RuntimeError("Runtime stream ended without a measured result")
                    try:
                        _check_result(result, size, config.output_tokens)
                    except Exception as exc:
                        report.failures.append(
                            {
                                **active_request,
                                "error": str(exc),
                                "result": result.model_dump(mode="json"),
                            }
                        )
                        raise
                    report.samples.append(
                        BenchmarkSample(
                            prompt_tokens=size,
                            repeat=repeat,
                            order=order,
                            result=result,
                        )
                    )
                    report.summary = summarize(report)
                    yield BenchmarkEvent(
                        kind="sample",
                        mode=mode,
                        repeat=repeat,
                        prompt_tokens=size,
                        message="Measured request complete",
                        report=report,
                    )
        report.status = "completed"
    except asyncio.CancelledError:
        report.status = "cancelled"
        await pipeline.cancel()
    except GeneratorExit:
        report.status = "cancelled"
        await pipeline.cancel()
        raise
    except Exception as exc:
        report.status = "cancelled" if cancelled() else "failed"
        report.errors.append(str(exc))
        if not report.failures:
            report.failures.append({**active_request, "error": str(exc)})
    finally:
        if report.status == "running":
            report.status = "cancelled"
        report.finished_at = datetime.now(UTC)
        report.summary = summarize(report)
        try:
            report.environment["health_after"] = await pipeline.health()
        except Exception as exc:
            report.environment["health_after_error"] = str(exc)
        await asyncio.to_thread(save_report, report, settings.reports_dir)
    kind = {"completed": "complete", "cancelled": "cancelled", "failed": "error"}[report.status]
    yield BenchmarkEvent(kind=kind, stage=report.status, message=report.status, report=report)


async def collect_benchmark(
    pipeline, config: BenchmarkConfig, settings, *, cancel_event: asyncio.Event | None = None
) -> BenchmarkReport:
    report = None
    async for event in run_benchmark(pipeline, config, settings, cancel_event=cancel_event):
        if event.report is not None:
            report = event.report
    if report is None:
        raise RuntimeError("Benchmark completed without a report")
    return report
