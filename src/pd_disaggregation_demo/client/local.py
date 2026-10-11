"""Typed client for explicitly reusing an already running Mac API."""

import contextlib
import json
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal, Self

import httpx

from ..benchmarks import BenchmarkConfig, BenchmarkEvent, BenchmarkReport
from ..models import (
    ArtifactManifest,
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    PrefillRequest,
    RuntimeProfile,
)


def _identifier(value: str) -> str:
    if re.fullmatch(r"[a-f0-9]{32}", value) is None:
        raise ValueError("Invalid artifact or benchmark job ID")
    return value


class LocalClient:
    def __init__(
        self,
        api_url: str,
        *,
        timeout: float = 1200,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        try:
            url = httpx.URL(api_url)
        except httpx.InvalidURL as exc:
            raise ValueError("Invalid --api-url HTTP(S) API URL") from exc
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.userinfo
            or url.fragment
            or url.query
        ):
            raise ValueError(
                "--api-url must be an HTTP(S) API URL without credentials, query, or fragment"
            )
        self.api_url = str(url).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.api_url + "/", timeout=timeout, trust_env=False, transport=transport
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _json(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        response = await self._client.request(method, path.lstrip("/"), json=data)
        response.raise_for_status()
        return response.json()

    async def configure(self, profile: RuntimeProfile) -> dict[str, Any]:
        return await self._json("POST", "/configure", profile.model_dump(mode="json"))

    async def health(self) -> dict[str, Any]:
        return await self._json("GET", "/health")

    async def cancel(self) -> dict[str, Any]:
        response = await self._client.post("cancel", json={}, timeout=10)
        response.raise_for_status()
        return response.json()

    async def prefill(self, request: PrefillRequest) -> ArtifactManifest:
        return ArtifactManifest.model_validate(
            await self._json("POST", "/prefill", request.model_dump(mode="json"))
        )

    async def manifest(self, artifact_id: str) -> ArtifactManifest:
        return ArtifactManifest.model_validate(
            await self._json("GET", f"/artifacts/{_identifier(artifact_id)}/manifest")
        )

    async def _events(
        self, method: str, path: str, data: dict[str, Any] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        async with self._client.stream(method, path.lstrip("/"), json=data) as response:
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                raise RuntimeError("API returned an unexpected non-SSE response")
            lines = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    lines.append(line[5:].lstrip(" "))
                elif line == "" and lines:
                    payload = "\n".join(lines)
                    lines = []
                    if payload == "[DONE]":
                        continue
                    value = json.loads(payload)
                    if not isinstance(value, dict):
                        raise RuntimeError("API SSE event must be a JSON object")
                    yield value
            if lines:
                raise RuntimeError("API SSE stream ended inside an incomplete event")

    async def _generation(self, path: str, data: dict[str, Any]) -> AsyncGenerator[GenerationEvent]:
        result_seen = False
        async with contextlib.aclosing(self._events("POST", path, data)) as events:
            async for value in events:
                if "error" in value and "kind" not in value:
                    event = GenerationEvent(kind="error", text=str(value["error"]))
                else:
                    event = GenerationEvent.model_validate(value)
                yield event
                if event.kind == "error":
                    raise RuntimeError(event.text or str(event.data) or "API generation failed")
                if event.kind == "result":
                    if event.result is None:
                        raise RuntimeError("API result event has no measured result")
                    result_seen = True
                    if not event.result.success:
                        raise RuntimeError(event.result.error or "API generation failed")
        if not result_seen:
            raise RuntimeError("API generation stream ended without a measured result")

    async def generate(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent]:
        async with contextlib.aclosing(
            self._generation("/generate", request.model_dump(mode="json"))
        ) as events:
            async for event in events:
                yield event

    async def decode(
        self, artifact_id: str, request: DecodeRequest
    ) -> AsyncGenerator[GenerationEvent]:
        data = {**request.model_dump(mode="json"), "artifact_id": _identifier(artifact_id)}
        async with contextlib.aclosing(self._generation("/decode", data)) as events:
            async for event in events:
                yield event

    async def benchmark(self, config: BenchmarkConfig) -> AsyncGenerator[BenchmarkEvent]:
        job = await self._json("POST", "/benchmarks", config.model_dump(mode="json"))
        job_id = _identifier(job["job_id"])
        terminal_seen = False
        async with contextlib.aclosing(
            self._events("GET", f"/benchmarks/{job_id}/events")
        ) as events:
            async for value in events:
                event = BenchmarkEvent.model_validate(value)
                terminal_seen |= event.kind in ("complete", "cancelled", "error")
                yield event
        status = await self._json("GET", f"/benchmarks/{job_id}")
        if not status.get("done"):
            raise RuntimeError("API benchmark event stream ended before its job completed")
        if status.get("report"):
            report = BenchmarkReport.model_validate(status["report"])
            if report.status == "running" and not terminal_seen:
                raise RuntimeError("API benchmark ended without a final report")
            if report.status != "running" and not terminal_seen:
                kind: dict[str, Literal["complete", "cancelled", "error"]] = {
                    "completed": "complete",
                    "cancelled": "cancelled",
                    "failed": "error",
                }
                yield BenchmarkEvent(kind=kind[report.status], stage=report.status, report=report)
        elif not terminal_seen:
            raise RuntimeError("API benchmark ended without a report or a terminal event")
