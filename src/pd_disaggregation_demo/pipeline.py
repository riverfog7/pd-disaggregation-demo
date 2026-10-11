"""End-to-end CUDA prefill -> native state transfer -> Metal decode."""

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator

from .artifacts import delete_artifact
from .client.remote import RemoteWorker
from .config.settings import Settings
from .models import (
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    PrefillArtifact,
    PrefillRequest,
    RuntimeProfile,
)
from .runtime.engine import LlamaRuntime, elapsed_ms


class Pipeline:
    def __init__(self, settings: Settings, profile: RuntimeProfile | None = None):
        self.settings = settings
        self.profile = profile or RuntimeProfile()
        self.runtime = LlamaRuntime(settings, self.profile)
        self.remote = RemoteWorker(settings)
        self._lock = asyncio.Lock()
        self._active: asyncio.Task | None = None
        self._remote_active = False

    @property
    def fingerprint(self):
        return self.runtime.fingerprint

    async def start(self) -> None:
        await self.runtime.start()

    async def close(self) -> None:
        await self.cancel()
        await self.runtime.close()
        await self.remote.close()

    async def configure(self, profile: RuntimeProfile) -> None:
        async with self._request():
            await self.runtime.configure(profile)
            self.profile = profile

    async def health(self) -> dict:
        local = await self.runtime.health()
        try:
            remote = await self.remote.health()
        except Exception as exc:
            remote = {"status": "unavailable", "error": str(exc)}
        return {**local, "local": local, "remote": remote}

    async def format_chat(self, messages: list[dict]) -> str:
        return await self.runtime.format_chat(messages)

    async def tokenize(self, prompt: str) -> list[int]:
        return await self.runtime.tokenize(prompt)

    @contextlib.asynccontextmanager
    async def session(self):
        """Own configuration and requests across one chat or benchmark sequence."""
        async with self._request():
            yield self

    @contextlib.asynccontextmanager
    async def _request(self):
        if self._lock.locked() and self._active is asyncio.current_task():
            yield
            return
        if self._lock.locked():
            raise RuntimeError("The demo supports one active request")
        async with self._lock:
            self._active = asyncio.current_task()
            try:
                yield
            finally:
                self._active = None

    async def cancel(self) -> None:
        active = self._active
        if active and active is not asyncio.current_task() and not active.done():
            active.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active
        # Worker cancellation is required after aborting a remote HTTP request;
        # disconnecting the worker API alone need not cancel native inference.
        if self._remote_active:
            await self.remote.cancel()
            self._remote_active = False

    async def _remote_prefill(self, tokens: list[int]) -> tuple[PrefillArtifact, float]:
        await self.remote.configure_matching(self.profile)
        manifest = None
        self._remote_active = True
        try:
            manifest = await self.remote.prefill(tokens)
            artifact, transfer_ms = await self.remote.download(manifest, self.fingerprint)
            return artifact, transfer_ms
        except BaseException:
            with contextlib.suppress(Exception):
                await self.remote.cancel()
            raise
        finally:
            self._remote_active = False
            if manifest is not None:
                await self.remote.delete(manifest.artifact_id)

    async def prefill(self, request: PrefillRequest) -> PrefillArtifact:
        async with self._request():
            tokens = request.token_ids
            if tokens is None:
                if request.prompt is None:
                    raise ValueError("Prefill needs prompt or canonical token_ids")
                tokens = await self.tokenize(request.prompt)
            if request.backend == "local":
                return await self.runtime.prefill(tokens)
            artifact, _ = await self._remote_prefill(tokens)
            return artifact

    async def decode(
        self, artifact: PrefillArtifact, request: DecodeRequest
    ) -> AsyncGenerator[GenerationEvent]:
        async with self._request():
            async with contextlib.aclosing(self.runtime.decode(artifact, request)) as stream:
                async for event in stream:
                    yield event

    async def generate(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent]:
        async with self._request():
            start = time.perf_counter()
            if request.mode == "local":
                async with contextlib.aclosing(self.runtime.generate(request)) as stream:
                    async for event in stream:
                        yield event
                return
            tokenize_start = time.perf_counter()
            tokens = request.token_ids
            if tokens is None:
                tokens = await self.tokenize(request.prompt)
            tokenize_ms = elapsed_ms(tokenize_start) if request.token_ids is None else 0
            if len(tokens) < 2 or len(tokens) + request.max_tokens > self.profile.context_size:
                raise ValueError("Split prompt needs at least two tokens and sufficient context")
            yield GenerationEvent(
                kind="stage", stage="remote_prefill", elapsed_ms=elapsed_ms(start)
            )
            artifact = None
            try:
                artifact, transfer_ms = await self._remote_prefill(tokens)
                yield GenerationEvent(
                    kind="stage",
                    stage="transfer",
                    elapsed_ms=elapsed_ms(start),
                    data={
                        "state_bytes": artifact.manifest.state_bytes,
                        "transfer_ms": transfer_ms,
                    },
                )
                decode_request = DecodeRequest.model_validate(
                    request.model_dump(
                        include={
                            "max_tokens",
                            "temperature",
                            "seed",
                            "ignore_eos",
                        }
                    )
                )
                first_token_ms = None
                async with contextlib.aclosing(
                    self.runtime.decode(artifact, decode_request)
                ) as stream:
                    async for event in stream:
                        event.elapsed_ms = elapsed_ms(start)
                        if event.kind == "token" and first_token_ms is None:
                            first_token_ms = event.elapsed_ms
                        if event.result:
                            result = event.result
                            result.mode = "split"
                            result.remote_profile = self.remote.profile
                            result.effective_settings["remote"] = self.remote.effective_settings
                            metrics = result.metrics
                            metrics.ttft_ms = first_token_ms
                            metrics.tokenize_ms = tokenize_ms
                            metrics.transfer_ms = transfer_ms
                            metrics.total_ms = elapsed_ms(start)
                            remote_peak = {
                                target: artifact.manifest.timings[source]
                                for source, target in [
                                    ("peak_process_rss_bytes", "process_rss_max_bytes"),
                                    ("peak_gpu_used_bytes", "gpu_used_max_bytes"),
                                    ("memory_sample_count", "sample_count"),
                                    ("memory_sample_period_ms", "sample_period_ms"),
                                ]
                                if source in artifact.manifest.timings
                            }
                            if remote_peak:
                                metrics.memory["remote_prefill_peak"] = remote_peak
                            if transfer_ms > 0:
                                metrics.transfer_mbps = (
                                    metrics.state_bytes * 8 / (transfer_ms * 1000)
                                )
                        yield event
            finally:
                if artifact:
                    await asyncio.to_thread(delete_artifact, artifact)
