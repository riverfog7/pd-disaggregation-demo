#!/usr/bin/env python3
"""Opt-in lifecycle checks against an existing, otherwise idle demo API.

Use --api-url http://127.0.0.1:8080; add --split and optionally --worker-url
http://127.0.0.1:8090 to include the configured CUDA worker. This starts only
HTTP generation requests, interrupts their streams, and checks fresh requests.
It never starts, configures, kills, or restarts engines, tunnels, or deployments.
The demo /cancel endpoint cancels its active request, so do not run other demo
clients concurrently. An observed SSE phase does not prove the exact internal
stage reached when the interruption arrives.

For an externally managed worker outage, pass --expect-worker-unavailable.
This requires an actionable split failure and successful local recovery. Add
--wait-for-worker-recovery SECONDS to wait for an external restart and require
split recovery too. Without that option, remote restart recovery is untested.
"""

import argparse
import asyncio
import contextlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from pd_disaggregation_demo.client.local import LocalClient
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import GenerationRequest, Mode

PROMPT = "Explain why the sky is blue in one sentence."


async def completion(client: LocalClient, mode: Mode) -> dict:
    ids = []
    result = None
    request = GenerationRequest(
        prompt=PROMPT, mode=mode, max_tokens=8, temperature=0, seed=123, ignore_eos=True
    )
    async with contextlib.aclosing(client.generate(request)) as source:
        async for event in source:
            ids.extend(event.data.get("token_ids", []))
            if event.result:
                result = event.result
    if result is None:
        raise AssertionError("Recovery request had no measured result")
    return {
        "text": result.text,
        "token_ids": ids,
        "generated_tokens": result.generated_tokens,
        "cache_n": result.metrics.cache_n,
        "prompt_n": result.metrics.prompt_n,
        "ttft_ms": result.metrics.ttft_ms,
        "total_ms": result.metrics.total_ms,
    }


async def recovery(client: LocalClient, mode: Mode, baseline: dict) -> dict:
    deadline = time.monotonic() + 15
    while True:
        try:
            result = await completion(client, mode)
            break
        except RuntimeError as error:
            if "active request" not in str(error) or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.1)
    if result["text"] != baseline["text"] or result["token_ids"] != baseline["token_ids"]:
        raise AssertionError("Fresh deterministic continuation differs after interruption")
    return result


async def interrupt(client: LocalClient, mode: Mode, action: str, delay: float) -> dict:
    started = time.perf_counter()
    observed = None
    response = None
    request = GenerationRequest(
        prompt=PROMPT, mode=mode, max_tokens=128, temperature=0, seed=123, ignore_eos=True
    )
    async with contextlib.aclosing(client.generate(request)) as source:
        async for event in source:
            if event.kind == "error":
                raise RuntimeError(event.text)
            if (mode == "local" and event.kind == "token") or (
                mode == "split" and event.stage == "remote_prefill"
            ):
                observed = {"kind": event.kind, "stage": event.stage}
                await asyncio.sleep(delay)
                if action == "cancel":
                    response = await client.cancel()
                break
    if observed is None:
        raise AssertionError(
            "Generation completed without reaching the requested interruption phase"
        )
    return {
        "action": action,
        "mode": mode,
        "observed_phase": observed,
        "delay_seconds": delay,
        "interruption_ms": (time.perf_counter() - started) * 1000,
        "cancel_response": response,
    }


def provenance(health: dict) -> dict:
    engine = health.get("local", health.get("engine", health))
    return {
        "status": health.get("status"),
        "git_revision": health.get("git_revision"),
        "profile": health.get("profile", engine.get("profile")),
        "engine_revision": engine.get("engine_revision"),
        "busy": health.get("busy"),
    }


async def worker_snapshot(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get(url.rstrip("/") + "/health")
            response.raise_for_status()
            return provenance(response.json())
    except httpx.HTTPError as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}


async def validate(
    api_url: str,
    *,
    split: bool = False,
    worker_url: str | None = None,
    expect_worker_unavailable: bool = False,
    wait_for_worker_recovery: float = 0,
    interruption_delay: float = 0.1,
    request_timeout: float | None = None,
) -> dict:
    request_timeout = Settings().request_timeout if request_timeout is None else request_timeout
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "api_url": api_url,
        "worker_url": worker_url,
        "request_timeout_seconds": request_timeout,
        "scope": "HTTP cancellation/disconnect and request recovery; no process control",
        "cases": [],
        "worker_failure": None,
        "success": False,
    }
    try:
        async with LocalClient(api_url, timeout=request_timeout) as client:
            health = await client.health()
            report["api_before"] = provenance(health)
            report["worker_before"] = provenance(health.get("remote", {}))
            if health.get("local", health).get("status") != "ready":
                raise AssertionError("The existing Mac API must have a ready native engine")
            if worker_url:
                report["worker_before"] = await worker_snapshot(worker_url)
                if report["worker_before"].get("busy"):
                    raise AssertionError("Worker is already busy; run against an idle demo")
            local = await completion(client, "local")
            if expect_worker_unavailable:
                if health.get("remote", {}).get("status") != "unavailable":
                    raise AssertionError(
                        "Expected the Mac API's configured worker to be unavailable"
                    )
                try:
                    await completion(client, "split")
                except RuntimeError as error:
                    message = str(error).strip()
                    if not message:
                        raise AssertionError("Worker failure had no actionable message") from error
                    report["worker_failure"] = {
                        "error": message,
                        "local_recovery": await recovery(client, "local", local),
                        "split_recovery": None,
                    }
                else:
                    raise AssertionError("Split generation succeeded during the expected outage")
                if wait_for_worker_recovery:
                    deadline = time.monotonic() + wait_for_worker_recovery
                    while True:
                        health = await client.health()
                        if health.get("remote", {}).get("status") == "ready":
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Worker did not recover before the requested deadline"
                            )
                        await asyncio.sleep(0.5)
                    report["worker_failure"]["split_recovery"] = await completion(client, "split")
                elif split:
                    raise ValueError("--split during an outage requires --wait-for-worker-recovery")
            baselines: dict[Mode, dict] = {"local": local}
            if split:
                baselines["split"] = await completion(client, "split")
            for mode, baseline in baselines.items():
                for action in ("cancel", "disconnect"):
                    case = await interrupt(client, mode, action, interruption_delay)
                    case["recovery"] = await recovery(client, mode, baseline)
                    report["cases"].append(case)
            health = await client.health()
            report["api_after"] = provenance(health)
            report["worker_after"] = (
                await worker_snapshot(worker_url)
                if worker_url
                else provenance(health.get("remote", {}))
            )
            report["success"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--worker-url")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--expect-worker-unavailable", action="store_true")
    parser.add_argument("--wait-for-worker-recovery", type=float, default=0)
    parser.add_argument("--interruption-delay", type=float, default=0.1)
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Inference/transfer timeout in seconds (default: Settings request_timeout)",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/api-lifecycle.json"))
    args = parser.parse_args()
    if not 0 <= args.interruption_delay <= 1 or args.wait_for_worker_recovery < 0:
        parser.error("Interruption delay must be 0..1 seconds and recovery wait nonnegative")
    if args.wait_for_worker_recovery and not args.expect_worker_unavailable:
        parser.error("--wait-for-worker-recovery requires --expect-worker-unavailable")
    if args.request_timeout is not None and (
        not math.isfinite(args.request_timeout) or args.request_timeout <= 0
    ):
        parser.error("--request-timeout must be a finite positive number")
    report = asyncio.run(
        validate(
            args.api_url,
            split=args.split,
            worker_url=args.worker_url,
            expect_worker_unavailable=args.expect_worker_unavailable,
            wait_for_worker_recovery=args.wait_for_worker_recovery,
            interruption_delay=args.interruption_delay,
            request_timeout=args.request_timeout,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"success": report["success"], "output": str(args.output.resolve())}))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
