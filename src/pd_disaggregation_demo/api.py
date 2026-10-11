"""Local HTTP API and the benchmark dashboard share one native Metal process."""

import asyncio
import contextlib
import json
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from .artifacts import load_artifact
from .catalog import MODELS
from .config import Settings
from .models import (
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    PrefillRequest,
    RuntimeProfile,
)
from .pipeline import Pipeline

BENCHMARK_REPLAY_LIMIT = 256
BENCHMARK_JOB_LIMIT = 16
BENCHMARK_TERMINAL_KINDS = {"complete", "cancelled", "error"}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0)
    seed: int = 0
    mode: Literal["local", "split"] = "local"


class ArtifactDecodeRequest(DecodeRequest):
    artifact_id: str


def _safe_id(value: str) -> str:
    if re.fullmatch(r"[a-f0-9]{32}", value) is None:
        raise HTTPException(400, "Invalid artifact or run ID")
    return value


def _sse(data: Any, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    value = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"{prefix}data: {value}\n\n"


async def _events(source: AsyncGenerator) -> AsyncIterator[str]:
    try:
        async with contextlib.aclosing(source):
            async for event in source:
                yield _sse(event.model_dump(mode="json"), event.kind)
    except asyncio.CancelledError:
        event = GenerationEvent(kind="error", stage="cancelled", text="Request cancelled")
        yield _sse(event.model_dump(mode="json"), "error")
    except Exception as exc:
        event = GenerationEvent(kind="error", text=str(exc))
        yield _sse(event.model_dump(mode="json"), "error")


def create_app(
    settings: Settings | None = None,
    profile: RuntimeProfile | None = None,
    *,
    pipeline: Pipeline | None = None,
    with_ui: bool = True,
) -> FastAPI:
    from .benchmarks import BenchmarkConfig, run_benchmark

    settings = settings or Settings()
    pipeline = pipeline or Pipeline(settings, profile)
    jobs: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await pipeline.start()
        try:
            yield
        finally:
            for job in jobs.values():
                if not job["task"].done():
                    job["task"].cancel()
            await asyncio.gather(*(job["task"] for job in jobs.values()), return_exceptions=True)
            await pipeline.close()

    app = FastAPI(title="Prefill / Decode Benchmark Demo", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/demo" if with_ui else "/docs")

    @app.get("/health")
    async def health():
        return {**await pipeline.health(), "git_revision": settings.git_revision}

    @app.get("/capabilities")
    async def capabilities():
        return {
            "modes": ["local", "split"],
            "operations": ["prefill", "decode", "generate", "benchmark", "chat"],
            "cache_types": ["f16", "q8_0"],
            "concurrency": 1,
            "models": [spec.model_dump() for spec in MODELS.values()],
            "profile": pipeline.profile.model_dump(),
        }

    @app.post("/configure")
    async def configure(value: RuntimeProfile):
        try:
            await pipeline.configure(value)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return await health()

    @app.post("/generate")
    async def generate(request: GenerationRequest):
        return StreamingResponse(
            _events(pipeline.generate(request)), media_type="text/event-stream"
        )

    @app.post("/prefill")
    async def prefill(request: PrefillRequest):
        try:
            artifact = await pipeline.prefill(request)
            return artifact.manifest
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    def artifact(identifier: str):
        directory = settings.artifacts_dir / _safe_id(identifier)
        if not directory.is_dir() or directory.is_symlink():
            raise HTTPException(404, "Artifact not found")
        try:
            return load_artifact(directory)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/artifacts/{artifact_id}/manifest")
    async def artifact_manifest(artifact_id: str):
        return (await asyncio.to_thread(artifact, artifact_id)).manifest

    @app.get("/artifacts/{artifact_id}/state")
    async def artifact_state(artifact_id: str):
        value = await asyncio.to_thread(artifact, artifact_id)
        return FileResponse(value.state_path, media_type="application/octet-stream")

    @app.post("/decode")
    async def decode(request: ArtifactDecodeRequest):
        value = await asyncio.to_thread(artifact, request.artifact_id)
        options = DecodeRequest.model_validate(request.model_dump(exclude={"artifact_id"}))
        return StreamingResponse(
            _events(pipeline.decode(value, options)), media_type="text/event-stream"
        )

    @app.post("/cancel")
    async def cancel():
        for job in jobs.values():
            job["cancel"].set()
        await pipeline.cancel()
        return {"status": "cancelled"}

    @app.post("/benchmarks", status_code=202)
    async def benchmark(config: BenchmarkConfig):
        if any(not job["task"].done() for job in jobs.values()):
            raise HTTPException(409, "A benchmark is already active")
        # Reports remain on disk; only the most recent HTTP job handles stay in memory.
        for old_id in list(jobs)[: max(0, len(jobs) - BENCHMARK_JOB_LIMIT + 1)]:
            del jobs[old_id]
        job_id = uuid4().hex
        job: dict[str, Any] = {
            "events": deque(maxlen=BENCHMARK_REPLAY_LIMIT),
            "event_count": 0,
            "report": None,
            "status": "running",
            "cancel": asyncio.Event(),
        }

        def remember(event):
            job["events"].append(event)
            job["event_count"] += 1

        async def run():
            try:
                async for event in run_benchmark(
                    pipeline, config, settings, cancel_event=job["cancel"]
                ):
                    if event.report:
                        job["report"] = event.report
                        job["status"] = event.report.status
                    # A single live report owns all samples. Replay stores no growing snapshots.
                    remember(event.model_dump(mode="json", exclude={"report"}))
            except asyncio.CancelledError:
                job["status"] = "cancelled"
                if job["report"] is not None:
                    job["report"].status = "cancelled"
                remember({"kind": "cancelled"})
            except Exception as exc:
                job["status"] = "failed"
                if job["report"] is not None:
                    job["report"].status = "failed"
                    job["report"].errors.append(str(exc))
                remember({"kind": "error", "message": str(exc)})

        job["task"] = asyncio.create_task(run())
        jobs[job_id] = job
        return {"job_id": job_id, "events_url": f"/benchmarks/{job_id}/events"}

    @app.get("/benchmarks")
    async def benchmark_history():
        reports = []
        for path in sorted(
            settings.reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            with contextlib.suppress(ValueError, OSError):
                reports.append(json.loads(path.read_text()))
        return reports

    @app.get("/benchmarks/{job_id}")
    async def benchmark_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown benchmark job")
        job = jobs[job_id]
        return {
            "done": job["task"].done(),
            "status": job["status"],
            "report": job["report"].model_dump(mode="json") if job["report"] is not None else None,
            "events": list(job["events"]),
            "events_dropped": job["event_count"] - len(job["events"]),
        }

    @app.get("/benchmarks/{job_id}/events")
    async def benchmark_events(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown benchmark job")
        job = jobs[job_id]

        async def stream():
            position = 0
            while True:
                retained = list(job["events"])
                first = job["event_count"] - len(retained)
                for index in range(max(0, position - first), len(retained)):
                    event = retained[index]
                    if event["kind"] in BENCHMARK_TERMINAL_KINDS and job["report"] is not None:
                        event = {**event, "report": job["report"].model_dump(mode="json")}
                    yield _sse(event, event["kind"])
                    position = first + index + 1
                if job["task"].done():
                    if position >= job["event_count"]:
                        break
                    continue
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/reports/{run_id}/{format}")
    async def export_report(run_id: str, format: Literal["json", "csv", "html"]):
        path: Path = settings.reports_dir / f"{_safe_id(run_id)}.{format}"
        if not path.is_file():
            raise HTTPException(404, "Report not found")
        return FileResponse(path, filename=path.name)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": pipeline.profile.model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest):
        if request.model and request.model != pipeline.profile.model_id:
            raise HTTPException(400, "Requested model is not currently loaded; use /configure")

        async def chat_events():
            async with pipeline.session():
                prompt = await pipeline.format_chat(
                    [item.model_dump() for item in request.messages]
                )
                generation = GenerationRequest(
                    prompt=prompt,
                    mode=request.mode,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    seed=request.seed,
                )
                async with contextlib.aclosing(pipeline.generate(generation)) as source:
                    async for event in source:
                        yield event

        identifier = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        def chunk(delta: dict, finish_reason=None):
            return {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": pipeline.profile.model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        if request.stream:

            async def stream_chat():
                yield _sse(chunk({"role": "assistant"}))
                try:
                    async with contextlib.aclosing(chat_events()) as source:
                        async for event in source:
                            if event.kind == "token":
                                yield _sse(chunk({"content": event.text}))
                            if event.result:
                                reason = (
                                    "length"
                                    if event.result.generated_tokens >= request.max_tokens
                                    else "stop"
                                )
                                yield _sse(chunk({}, reason))
                except asyncio.CancelledError:
                    yield _sse({"error": {"message": "Request cancelled", "type": "cancelled"}})
                except Exception as exc:
                    yield _sse({"error": {"message": str(exc), "type": "runtime_error"}})
                yield _sse("[DONE]")

            return StreamingResponse(stream_chat(), media_type="text/event-stream")
        result = None
        async with contextlib.aclosing(chat_events()) as source:
            async for event in source:
                if event.result:
                    result = event.result
        if result is None:
            raise HTTPException(500, "Generation produced no result")
        return {
            "id": identifier,
            "object": "chat.completion",
            "created": created,
            "model": pipeline.profile.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "length"
                    if result.generated_tokens >= request.max_tokens
                    else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.prompt_tokens + result.generated_tokens,
            },
            "pd_metrics": result.metrics.model_dump(),
        }

    if with_ui:
        import gradio as gr

        from .ui.app import build_ui

        app = gr.mount_gradio_app(app, build_ui(pipeline, settings), path="/demo")
    return app
