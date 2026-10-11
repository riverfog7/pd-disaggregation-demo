"""Command line entry points for native Mac, Docker Linux, and measured operations."""

import asyncio
import contextlib
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import httpx
import psutil
import typer

from .catalog import MODELS, ensure_model
from .config import Settings
from .models import (
    ArtifactManifest,
    DecodeRequest,
    GenerationRequest,
    PrefillRequest,
    RuntimeProfile,
)

app = typer.Typer(
    no_args_is_help=True, help="Benchmark native Mac decode with local or CUDA prefill."
)
serve = typer.Typer(no_args_is_help=True)
models = typer.Typer(no_args_is_help=True)
app.add_typer(serve, name="serve")
app.add_typer(models, name="models")
ModelOption = Annotated[str, typer.Option("--model", help="Pinned GGUF model profile")]
ContextOption = Annotated[int, typer.Option("--context", min=256)]
CacheOption = Annotated[str, typer.Option("--cache", help="f16 or q8_0")]
ApiOption = Annotated[
    str | None, typer.Option("--api-url", help="Reuse an already running Mac API; starts no engine")
]


def _settings(**overrides) -> Settings:
    settings = Settings(**{key: value for key, value in overrides.items() if value is not None})
    if settings.git_revision == "unknown":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            settings.git_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
    return settings


def _profile(model: str, context: int, cache: str, **kwargs) -> RuntimeProfile:
    return RuntimeProfile.model_validate(
        dict(model_id=model, context_size=context, cache_type=cache, **kwargs)
    )


def _json(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, indent=2, default=str))


def _input(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise typer.BadParameter("Use --prompt or --prompt-file, not both")
    if prompt_file is not None:
        return prompt_file.read_text()
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("Provide --prompt, --prompt-file, or piped input")


def _pid_file(settings: Settings) -> Path:
    return settings.work_dir / "mac-server.json"


def _owned_process(settings: Settings):
    try:
        record = json.loads(_pid_file(settings).read_text())
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["created"]) > 0.01:
            return None
        return process
    except (OSError, ValueError, psutil.Error, KeyError):
        return None


@serve.command("mac")
def serve_mac(
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    port: int | None = None,
    detach: bool = False,
    tunnel: Annotated[bool, typer.Option("--tunnel/--no-tunnel")] = True,
    ui: Annotated[bool, typer.Option("--ui/--no-ui")] = True,
    lan_host: str | None = None,
):
    """Start native Metal inference, local API and the web dashboard at /demo."""
    if platform.system() != "Darwin":
        raise typer.BadParameter("Use serve linux for the Docker CUDA worker")
    settings = _settings(port=port, lan_ssh_host=lan_host)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if _owned_process(settings):
        raise typer.BadParameter("The managed Mac server is already running")
    if detach:
        command = [
            sys.executable,
            "-m",
            "pd_disaggregation_demo.cli",
            "serve",
            "mac",
            "--model",
            model,
            "--context",
            str(context),
            "--cache",
            cache,
            "--port",
            str(settings.port),
            "--tunnel" if tunnel else "--no-tunnel",
            "--ui" if ui else "--no-ui",
        ]
        if lan_host:
            command.extend(["--lan-host", lan_host])
        with (settings.work_dir / "mac-server.log").open("ab") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        _json(
            {
                "pid": process.pid,
                "url": f"http://127.0.0.1:{settings.port}/demo",
                "log": str(settings.work_dir / "mac-server.log"),
                "status": "starting",
            }
        )
        return
    import uvicorn

    from .api import create_app
    from .deployment import open_tunnel

    process = psutil.Process()
    _pid_file(settings).write_text(
        json.dumps({"pid": process.pid, "created": process.create_time()})
    )
    forwarding = None
    try:
        if tunnel:
            forwarding = open_tunnel(
                settings, ssh_host=settings.lan_ssh_host or settings.remote_ssh_host
            )
        uvicorn.run(
            create_app(settings, _profile(model, context, cache), with_ui=ui),
            host=settings.host,
            port=settings.port,
        )
    finally:
        if forwarding and forwarding.poll() is None:
            forwarding.terminate()
            forwarding.wait(timeout=10)
        _pid_file(settings).unlink(missing_ok=True)


@serve.command("linux")
def serve_linux_command(
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    detach: Annotated[bool, typer.Option("--detach/--foreground")] = True,
    cpu_ffn: int = 0,
    ubatch: int = 128,
    batch: int = 512,
):
    """Build and launch the CUDA prefill worker through Docker Compose."""
    from .deployment import serve_linux

    _json(
        serve_linux(
            _settings(),
            _profile(model, context, cache, cpu_ffn=cpu_ffn, ubatch_size=ubatch, batch_size=batch),
            detach,
        )
    )


@app.command(hidden=True)
def worker():
    """Container entry point; not a host-native Linux service."""
    import uvicorn

    from .backend.app import create_worker_app

    settings = _settings()
    profile = RuntimeProfile.model_validate_json(os.environ.get("PDD_WORKER_PROFILE") or "{}")
    uvicorn.run(create_worker_app(settings, profile), host="0.0.0.0", port=settings.worker_port)


@app.command()
def setup(model: ModelOption = "qwen3.5-4b"):
    """Fetch and verify weights; build the pinned native engine on macOS."""
    from .runtime.build import ensure_engine

    settings = _settings()
    path = ensure_model(settings, model)
    result = {"model": str(path)}
    if platform.system() == "Darwin":
        result["engine"] = str(ensure_engine(settings, "metal"))
    else:
        result["next"] = "pd-demo serve linux --detach (builds CUDA inside Docker)"
    _json(result)


@models.command("list")
def list_models():
    _json([model.model_dump() for model in MODELS.values()])


@models.command("fetch")
def fetch_model(model: str = "qwen3.5-4b"):
    _json({"path": str(ensure_model(_settings(), model))})


@app.command()
def doctor(
    remote: bool = False,
    model: ModelOption = "qwen3.5-0.8b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
):
    """Read resources and admission; --remote also inspects SSH, GPU and Docker."""
    from .catalog import get_model
    from .deployment import inspect_remote
    from .resources import capture_resources, check_admission

    settings = _settings()
    backend = "metal" if platform.system() == "Darwin" else "cuda"
    result = {
        "resources": capture_resources(settings.work_dir).model_dump(mode="json"),
        "admission": check_admission(
            settings, _profile(model, context, cache), get_model(model), backend
        ).model_dump(mode="json"),
    }
    if remote:
        result["remote"] = inspect_remote(settings)
    _json(result)


@app.command()
def status(side: Annotated[str, typer.Argument()] = "mac"):
    """Inspect the local Mac service, local Docker worker, or remote host."""
    from .deployment import monitor, status_linux

    settings = _settings()
    if side == "linux":
        _json(status_linux(settings))
    elif side == "remote":
        _json(monitor(settings))
    elif side == "mac":
        try:
            response = httpx.get(f"http://127.0.0.1:{settings.port}/health", timeout=10)
            response.raise_for_status()
            _json(response.json())
        except httpx.HTTPError as exc:
            _json({"status": "unavailable", "error": str(exc)})
    else:
        raise typer.BadParameter("Choose mac, linux or remote")


@app.command()
def stop(side: Annotated[str, typer.Argument()] = "mac"):
    """Stop only the server owned by this demo."""
    from .deployment import stop_linux, stop_remote

    settings = _settings()
    if side == "linux":
        _json(stop_linux(settings))
    elif side == "remote":
        _json(stop_remote(settings))
    elif side == "mac":
        process = _owned_process(settings)
        if process is not None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired as exc:
                raise typer.BadParameter("Mac server did not stop within 30 seconds") from exc
        _pid_file(settings).unlink(missing_ok=True)
        _json({"status": "stopped"})
    else:
        raise typer.BadParameter("Choose mac, linux or remote")


@app.command()
def deploy(
    ref: Annotated[str, typer.Option("--ref")],
    host: str = "gpu_home",
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    cpu_ffn: int = 0,
    ubatch: int = 128,
):
    """Push/fetch a committed SHA through Git and verify its Docker release."""
    from .deployment import deploy as deploy_release

    _json(
        deploy_release(
            _settings(remote_ssh_host=host),
            ref,
            profile=_profile(model, context, cache, cpu_ffn=cpu_ffn, ubatch_size=ubatch),
        )
    )


@app.command()
def rollback(host: str = "gpu_home"):
    from .deployment import rollback as rollback_release

    _json(rollback_release(_settings(remote_ssh_host=host)))


async def _operation(settings, profile, operation):
    from .pipeline import Pipeline

    pipeline = Pipeline(settings, profile)
    try:
        await pipeline.start()
        return await operation(pipeline)
    finally:
        await pipeline.close()


async def _api_operation(api_url, settings, profile, operation):
    from .client.local import LocalClient

    async with LocalClient(api_url, timeout=settings.request_timeout) as client:
        try:
            if profile is not None:
                await client.configure(profile)
            return await operation(client)
        except asyncio.CancelledError:
            with contextlib.suppress(httpx.HTTPError, RuntimeError):
                await client.cancel()
            raise


@app.command()
def prefill(
    prompt: str | None = None,
    prompt_file: Path | None = None,
    remote: bool = False,
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    api_url: ApiOption = None,
):
    """Save a local or remote prefill artifact on this Mac."""
    text = _input(prompt, prompt_file)

    if api_url is not None:

        async def api_run(client):
            manifest = await client.prefill(
                PrefillRequest(prompt=text, backend="remote" if remote else "local")
            )
            _json(
                {
                    "artifact_id": manifest.artifact_id,
                    "api_url": client.api_url,
                    "manifest": manifest.model_dump(mode="json"),
                }
            )

        asyncio.run(_api_operation(api_url, _settings(), _profile(model, context, cache), api_run))
        return

    async def run(pipeline):
        value = await pipeline.prefill(
            PrefillRequest(prompt=text, backend="remote" if remote else "local")
        )
        _json(
            {
                "directory": str(value.state_path.parent),
                "manifest": value.manifest.model_dump(mode="json"),
            }
        )

    asyncio.run(_operation(_settings(), _profile(model, context, cache), run))


@app.command()
def decode(artifact: Path, max_tokens: int = 128, api_url: ApiOption = None):
    """Resume the matching model/cache/context from a saved prefill directory."""
    from .artifacts import load_artifact

    if api_url is not None:
        manifest_path = artifact if artifact.is_file() else artifact / "manifest.json"
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    else:
        value = load_artifact(artifact)
        manifest = value.manifest
    fingerprint = manifest.fingerprint
    profile = RuntimeProfile(
        model_id=manifest.model_id,
        context_size=fingerprint.context_size,
        cache_type=fingerprint.cache_type_k,
    )

    if api_url is not None:

        async def api_run(client):
            server_manifest = await client.manifest(manifest.artifact_id)
            if server_manifest != manifest:
                raise RuntimeError(
                    "Server artifact manifest differs from the selected local manifest"
                )
            async with contextlib.aclosing(
                client.decode(manifest.artifact_id, DecodeRequest(max_tokens=max_tokens))
            ) as events:
                async for event in events:
                    typer.echo(event.model_dump_json())

        asyncio.run(_api_operation(api_url, _settings(), profile, api_run))
        return

    async def run(pipeline):
        async for event in pipeline.decode(value, DecodeRequest(max_tokens=max_tokens)):
            typer.echo(event.model_dump_json())

    asyncio.run(_operation(_settings(), profile, run))


@app.command()
def generate(
    prompt: str | None = None,
    prompt_file: Path | None = None,
    mode: str = "local",
    max_tokens: int = 128,
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    api_url: ApiOption = None,
):
    """Generate with newline-delimited typed progress and measurements."""
    request = GenerationRequest.model_validate(
        dict(prompt=_input(prompt, prompt_file), mode=mode, max_tokens=max_tokens)
    )

    async def run(pipeline):
        async with contextlib.aclosing(pipeline.generate(request)) as events:
            async for event in events:
                typer.echo(event.model_dump_json())

    if api_url is not None:
        asyncio.run(_api_operation(api_url, _settings(), _profile(model, context, cache), run))
    else:
        asyncio.run(_operation(_settings(), _profile(model, context, cache), run))


@app.command()
def benchmark(
    model: ModelOption = "qwen3.5-4b",
    context: ContextOption = 8192,
    cache: CacheOption = "f16",
    sizes: str = "1024,4096,8192,16384,32768",
    repeats: int = 5,
    output_tokens: int = 128,
    api_url: ApiOption = None,
):
    """Run alternating local/split pairs and save JSON, CSV and HTML reports."""
    from .benchmarks import BenchmarkConfig, run_benchmark

    settings = _settings()
    config = BenchmarkConfig(
        profile=_profile(model, context, cache),
        prompt_sizes=[int(value) for value in sizes.split(",")],
        repeats=repeats,
        output_tokens=output_tokens,
    )
    status = "failed"

    async def run(pipeline):
        nonlocal status
        async for event in run_benchmark(pipeline, config, settings):
            typer.echo(event.model_dump_json())
            if event.report:
                status = event.report.status

    if api_url is not None:

        async def api_run(client):
            nonlocal status
            async with contextlib.aclosing(client.benchmark(config)) as events:
                async for event in events:
                    typer.echo(event.model_dump_json())
                    if event.report:
                        status = event.report.status
                    elif event.kind in ("error", "cancelled"):
                        status = "failed" if event.kind == "error" else "cancelled"

        asyncio.run(_api_operation(api_url, settings, None, api_run))
    else:
        asyncio.run(_operation(settings, config.profile, run))
    if status != "completed":
        raise typer.Exit(1)


def main():
    try:
        app()
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
