#!/usr/bin/env python3
"""Opt-in native hardware acceptance; never run as part of default pytest.

Run locally with `uv run python scripts/validate_handoff.py`. Add
`--worker-url http://127.0.0.1:8090` (for an SSH-forwarded worker), or an explicitly
accessible worker URL, to validate CUDA export -> Metal restore as well. The local
native server uses a private port and is always closed. Settings also honor PDD_*
environment variables. Add `--prompt-tokens 8192` for an exact benchmark corpus
prompt and an automatically sized context. This validates a specific run, not universal numerical
identity between kernels, cache formats, or devices.
"""

import argparse
import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pd_disaggregation_demo.artifacts import delete_artifact
from pd_disaggregation_demo.client.remote import RemoteWorker
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import (
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    PrefillArtifact,
    RuntimeProfile,
)
from pd_disaggregation_demo.runtime.engine import LlamaRuntime

PROMPT = "Write one short sentence explaining why the sky appears blue."
METRIC_KEYS = (
    "ttft_ms",
    "prefill_ms",
    "export_ms",
    "transfer_ms",
    "restore_ms",
    "final_prompt_ms",
    "decode_ms",
    "total_ms",
    "prefill_tps",
    "decode_tps",
    "client_decode_tps",
    "state_bytes",
    "cache_n",
    "prompt_n",
)


async def collect(source: AsyncGenerator[GenerationEvent]) -> dict:
    token_ids = []
    result = None
    try:
        async for event in source:
            if event.kind == "token":
                token_ids.extend(event.data.get("token_ids", []))
            if event.result:
                result = event.result
    finally:
        await source.aclose()
    if result is None:
        raise RuntimeError("Native generation produced no final result")
    metrics = result.metrics.model_dump(mode="json", include=set(METRIC_KEYS))
    metrics["memory_peak"] = result.metrics.memory.get("peak", {})
    return {"text": result.text, "token_ids": token_ids, "metrics": metrics}


def comparison(expected: dict, actual: dict) -> dict:
    left, right = expected["token_ids"], actual["token_ids"]
    first = next((i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b), None)
    if first is None and len(left) != len(right):
        first = min(len(left), len(right))
    return {
        "same_text": expected["text"] == actual["text"],
        "same_tokens": left == right,
        "first_token_difference": first,
        "expected_tokens": len(left),
        "actual_tokens": len(right),
    }


def require_same_continuation(expected: dict, actual: dict, label: str) -> dict:
    result = comparison(expected, actual)
    if not result["same_text"] or not result["same_tokens"]:
        raise AssertionError(f"{label}: deterministic continuation differs ({result})")
    return result


def artifact_summary(artifact: PrefillArtifact) -> dict:
    m = artifact.manifest
    return {
        "source_backend": m.source_backend,
        "prefix_tokens": m.prefix_tokens,
        "state_bytes": m.state_bytes,
        "state_sha256": m.state_sha256,
        "fingerprint": m.fingerprint.model_dump(mode="json"),
        "timings": m.timings,
    }


async def rejection_checks(runtime, artifact, request, expected):
    outcomes = []
    negative_dir = artifact.state_path.parent / "negative-tests"
    negative_dir.mkdir()
    try:
        for kind in ("corrupt", "truncated", "missing", "expired", "incompatible"):
            state = negative_dir / f"{kind}.bin"
            await asyncio.to_thread(shutil.copyfile, artifact.state_path, state)
            manifest = artifact.manifest.model_copy(deep=True)
            if kind == "corrupt":
                with state.open("r+b") as target:
                    byte = target.read(1)
                    target.seek(0)
                    target.write(bytes([byte[0] ^ 0xFF]))
            elif kind == "truncated":
                with state.open("r+b") as target:
                    target.truncate(manifest.state_bytes - 1)
            elif kind == "missing":
                state.unlink()
            elif kind == "expired":
                manifest.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            else:
                manifest.fingerprint = manifest.fingerprint.model_copy(
                    update={
                        "context_size": manifest.fingerprint.context_size + 256,
                    }
                )
            invalid = PrefillArtifact(manifest=manifest, state_path=state)
            rejection = None
            try:
                await collect(runtime.decode(invalid, request))
            except ValueError as error:
                rejection = str(error)
            if rejection is None:
                raise AssertionError(f"Native decode accepted {kind} artifact")
            recovered = await collect(runtime.decode(artifact, request))
            if (
                recovered["metrics"]["cache_n"] != artifact.manifest.prefix_tokens
                or recovered["metrics"]["prompt_n"] != 1
            ):
                raise AssertionError("Runtime did not remain usable after artifact rejection")
            recovered_comparison = require_same_continuation(
                expected, recovered, f"Recovery after {kind} artifact rejection"
            )
            outcomes.append(
                {
                    "case": kind,
                    "rejected": True,
                    "reason": rejection,
                    "runtime_usable_afterward": True,
                    "restored_cache_n": recovered["metrics"]["cache_n"],
                    "restored_prompt_n": recovered["metrics"]["prompt_n"],
                    "comparison": recovered_comparison,
                }
            )
    finally:
        shutil.rmtree(negative_dir, ignore_errors=True)
    return outcomes


async def validate(
    settings: Settings,
    model_id: str,
    max_tokens: int,
    use_worker: bool,
    prompt_tokens: int | None = None,
) -> dict:
    context_size = 1024
    if prompt_tokens is not None:
        if not 2 <= prompt_tokens <= 262144 or prompt_tokens + max_tokens > 262144:
            raise ValueError(
                "Prompt tokens plus output must fit the supported 262144-token context"
            )
        context_size = max(1024, ((prompt_tokens + max_tokens + 255) // 256) * 256)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "engine_revision": settings.engine_revision,
        "model_id": model_id,
        "scope": (
            "specific deterministic native runs; same-backend continuations must agree; "
            "cross-device and cross-cache numerical comparisons are observations"
        ),
        "prompt": PROMPT if prompt_tokens is None else "benchmark exact_tokens corpus",
        "requested_prompt_tokens": prompt_tokens,
        "context_size": context_size,
        "max_tokens": max_tokens,
        "local": [],
        "cuda_to_metal": [],
        "cross_cache_comparison": None,
        "success": False,
    }
    remote = RemoteWorker(settings) if use_worker else None
    direct_runs = []
    try:
        for cache in ("f16", "q8_0"):
            runtime = LlamaRuntime(
                settings,
                RuntimeProfile(
                    model_id=model_id,
                    cache_type=cache,
                    context_size=context_size,
                    batch_size=256,
                    ubatch_size=128,
                ),
                backend="metal",
            )
            artifact = None
            remote_artifact = None
            remote_manifest = None
            failure = None
            try:
                await runtime.start()
                if prompt_tokens is None:
                    prompt = await runtime.format_chat([{"role": "user", "content": PROMPT}])
                    token_ids = await runtime.tokenize(prompt)
                else:
                    from pd_disaggregation_demo.benchmarks.runner import exact_tokens

                    token_ids = await exact_tokens(runtime, prompt_tokens)
                    if len(token_ids) != prompt_tokens:
                        raise AssertionError(
                            "Benchmark corpus did not yield the requested token count"
                        )
                request = DecodeRequest(
                    max_tokens=max_tokens, temperature=0, seed=123, ignore_eos=True
                )
                direct = await collect(
                    runtime.generate(
                        GenerationRequest(
                            token_ids=token_ids,
                            max_tokens=max_tokens,
                            temperature=0,
                            seed=123,
                            ignore_eos=True,
                        )
                    )
                )
                direct_runs.append(direct)
                artifact = await runtime.prefill(token_ids)
                repeated = [await collect(runtime.decode(artifact, request)) for _ in range(2)]
                report["local"].append(
                    {
                        "cache_type": cache,
                        "startup_ms": runtime.startup_ms,
                        "canonical_token_ids": token_ids,
                        "prompt_tokens": len(token_ids),
                        "requested_prompt_tokens": prompt_tokens,
                        "actual_canonical_tokens": len(token_ids),
                        "artifact": artifact_summary(artifact),
                        "direct": direct,
                        "same_artifact_restored_twice": [
                            {
                                **item,
                                "comparison_with_direct": comparison(direct, item),
                            }
                            for item in repeated
                        ],
                        "repeat_comparison": comparison(repeated[0], repeated[1]),
                        "rejections": [],
                    }
                )
                for index, item in enumerate(repeated, start=1):
                    require_same_continuation(direct, item, f"{cache} local restore {index}")
                require_same_continuation(
                    repeated[0], repeated[1], f"{cache} repeated local restores"
                )
                report["local"][-1]["rejections"] = await rejection_checks(
                    runtime, artifact, request, repeated[0]
                )
                if remote is not None:
                    profile = await remote.configure_matching(runtime.profile)
                    try:
                        remote_manifest = await remote.prefill(token_ids)
                    except BaseException:
                        with contextlib.suppress(Exception):
                            await remote.cancel()
                        raise
                    remote_artifact, transfer_ms = await remote.download(
                        remote_manifest, runtime.fingerprint
                    )
                    decoded = [
                        await collect(runtime.decode(remote_artifact, request)) for _ in range(2)
                    ]
                    for item in decoded:
                        if (
                            item["metrics"]["cache_n"] != len(token_ids) - 1
                            or item["metrics"]["prompt_n"] != 1
                        ):
                            raise AssertionError("CUDA->Metal state was silently recomputed")
                    report["cuda_to_metal"].append(
                        {
                            "cache_type": cache,
                            "requested_prompt_tokens": prompt_tokens,
                            "actual_canonical_tokens": len(token_ids),
                            "remote_profile": profile.model_dump(mode="json"),
                            "artifact": artifact_summary(remote_artifact),
                            "transfer_ms": transfer_ms,
                            "restored_twice": [
                                {
                                    **item,
                                    "comparison_with_metal_direct": comparison(direct, item),
                                }
                                for item in decoded
                            ],
                            "repeat_comparison": comparison(decoded[0], decoded[1]),
                            "rejections": [],
                        }
                    )
                    require_same_continuation(
                        decoded[0], decoded[1], f"{cache} repeated CUDA artifact Metal restores"
                    )
                    report["cuda_to_metal"][-1]["rejections"] = await rejection_checks(
                        runtime, remote_artifact, request, decoded[0]
                    )
            except BaseException as error:
                failure = error
                raise
            finally:
                if artifact:
                    delete_artifact(artifact)
                if remote_artifact:
                    delete_artifact(remote_artifact)
                try:
                    await runtime.close()
                except Exception as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Native engine cleanup failed: {cleanup_error}")
                if remote and remote_manifest:
                    try:
                        await remote.delete(remote_manifest.artifact_id)
                    except Exception as cleanup_error:
                        if failure is None:
                            raise
                        failure.add_note(f"Remote artifact cleanup failed: {cleanup_error}")
        report["cross_cache_comparison"] = comparison(direct_runs[0], direct_runs[1])
        report["success"] = True
        return report
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "cleanup_notes": getattr(error, "__notes__", []),
        }
        return report
    finally:
        if remote:
            await remote.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--engine-port", type=int, default=8181)
    parser.add_argument("--worker-url", help="Opt in to CUDA worker state export and Metal restore")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument(
        "--prompt-tokens", type=int, help="Use exactly N canonical benchmark corpus tokens"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192:
        parser.error("--max-tokens must be between 1 and 8192")
    if args.prompt_tokens is not None and (
        not 2 <= args.prompt_tokens <= 262144 or args.prompt_tokens + args.max_tokens > 262144
    ):
        parser.error(
            "--prompt-tokens must be at least 2 and prompt plus output must not exceed 262144"
        )
    overrides = {"engine_port": args.engine_port}
    for field in ("work_dir", "cache_dir"):
        if (value := getattr(args, field)) is not None:
            overrides[field] = value
    if args.worker_url:
        overrides["remote_url"] = args.worker_url
    settings = Settings(**overrides)
    settings = settings.model_copy(
        update={"work_dir": settings.work_dir / f"validation-{uuid4().hex}"}
    )
    output = args.output or settings.reports_dir / "handoff-validation.json"
    report = asyncio.run(
        validate(
            settings,
            args.model,
            args.max_tokens,
            args.worker_url is not None,
            args.prompt_tokens,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "success": report["success"],
                "output": str(output.resolve()),
                "local_cache_types": [item["cache_type"] for item in report["local"]],
                "cuda_to_metal_cache_types": [
                    item["cache_type"] for item in report["cuda_to_metal"]
                ],
                "cross_cache_comparison": report["cross_cache_comparison"],
            }
        )
    )

    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
