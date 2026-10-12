"""Run the existing paired benchmark with explicit long-context experiment settings."""

import argparse
import asyncio
import contextlib
import json
import platform
import signal
import subprocess
from pathlib import Path

from pd_disaggregation_demo.benchmarks.models import BenchmarkConfig
from pd_disaggregation_demo.benchmarks.runner import run_benchmark
from pd_disaggregation_demo.calibration import atomic_report
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.deployment import open_tunnel
from pd_disaggregation_demo.models import RuntimeProfile
from pd_disaggregation_demo.pipeline import Pipeline


async def run(args) -> int:
    if platform.system() != "Darwin":
        raise ValueError("The paired benchmark runs on the Mac; CUDA stays in its guarded worker")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    settings = Settings(
        work_dir=args.work_dir.resolve(),
        engine_port=args.engine_port,
        git_revision=revision,
        remote_url=args.remote_url,
        request_timeout=args.timeout,
        startup_timeout=600,
        allow_metal_memory_overcommit=args.allow_metal_memory_overcommit,
    )
    profile = RuntimeProfile(
        model_id=args.model,
        cache_type="q8_0",
        context_size=args.tokens + args.output_tokens,
        batch_size=args.tokens,
        ubatch_size=args.ubatch,
        threads=args.threads,
    )
    config = BenchmarkConfig(
        profile=profile,
        prompt_sizes=[args.tokens],
        output_tokens=args.output_tokens,
        repeats=args.repeats,
        label=f"{args.model} · {args.tokens:,} tokens · actual SSH route",
    )
    pipeline = Pipeline(settings, profile)
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    tunnel = None
    report = None
    try:
        if args.ssh_host:
            tunnel = open_tunnel(settings, ssh_host=args.ssh_host)
        async with contextlib.aclosing(run_benchmark(pipeline, config, settings)) as events:
            async for event in events:
                print(
                    json.dumps(
                        {
                            "kind": event.kind,
                            "stage": event.stage,
                            "message": event.message,
                            "mode": event.mode,
                            "repeat": event.repeat,
                        }
                    ),
                    flush=True,
                )
                if event.report is not None:
                    report = event.report
                    report.environment["metal_memory_overcommit"] = (
                        args.allow_metal_memory_overcommit
                    )
                    report.environment["experiment_command"] = vars(args) | {
                        "work_dir": str(args.work_dir),
                        "output": str(args.output),
                    }
                    atomic_report(args.output, report.model_dump(mode="json"))
                if event.kind == "sample" and report:
                    print(report.samples[-1].result.metrics.model_dump_json(), flush=True)
        if report is None:
            raise RuntimeError("Benchmark produced no report")
        # Persist the experiment metadata into the normal JSON/CSV/HTML exports too.
        from pd_disaggregation_demo.benchmarks.storage import save_report

        save_report(report, settings.reports_dir)
        atomic_report(args.output, report.model_dump(mode="json"))
        print(json.dumps({"status": report.status, "exports": report.exports}), flush=True)
        return 0 if report.status == "completed" and report.samples else 1
    finally:
        await pipeline.close()
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
            await asyncio.to_thread(tunnel.wait, 10)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3.5-4b", "qwen3.5-27b"], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--engine-port", type=int, default=8181)
    parser.add_argument("--remote-url", default="http://127.0.0.1:8090")
    parser.add_argument("--ssh-host", default="gpu_home")
    parser.add_argument("--allow-metal-memory-overcommit", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.tokens <= 262016:
        parser.error("tokens must be between 2 and 262016")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
