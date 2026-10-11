"""One native llama-server slot, including hybrid recurrent sequence state."""

import asyncio
import contextlib
import json
import platform
import shutil
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import anyio
import httpx
import psutil

from ..artifacts import ensure_state_disk_space, sha256_file, validate_artifact, write_manifest
from ..config.settings import Settings
from ..models import (
    ArtifactManifest,
    Backend,
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    GenerationResult,
    PrefillArtifact,
    RunMetrics,
    RuntimeFingerprint,
    RuntimeProfile,
)
from .build import ensure_engine


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class AdmissionError(RuntimeError):
    """Requested model/profile exceeds the measured memory or disk reserves."""


class LlamaRuntime:
    def __init__(
        self,
        settings: Settings,
        profile: RuntimeProfile | None = None,
        backend: Backend | None = None,
    ):
        self.settings = settings
        self.profile = profile or RuntimeProfile()
        self.backend: Backend = backend or ("metal" if platform.system() == "Darwin" else "cuda")
        self.fingerprint: RuntimeFingerprint | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._active: asyncio.Task | None = None
        self._log = None
        self._slots_dir = settings.work_dir / "native-slots"
        self._admission = None
        self.startup_ms = 0.0
        self._peak_memory: dict = {}
        self._native_props: dict | None = None
        self._effective_settings: dict | None = None

    def _arguments(self, binary: Path, model_path: Path) -> list[str]:
        p = self.profile
        args = [
            str(binary),
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.settings.engine_port),
            "--parallel",
            "1",
            "--ctx-size",
            str(p.context_size),
            "--batch-size",
            str(p.batch_size),
            "--ubatch-size",
            str(p.ubatch_size),
            "--threads",
            str(p.threads),
            "--threads-batch",
            str(p.threads),
            "--gpu-layers",
            str(p.gpu_layers),
            "--n-cpu-ffn",
            str(p.cpu_ffn),
            "--cache-type-k",
            p.cache_type,
            "--cache-type-v",
            p.cache_type,
            "--flash-attn",
            "on",
            "--fit",
            "off",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "--no-repack",
            "--no-host",
            "--load-mode",
            "mmap",
            "--slot-save-path",
            str(self._slots_dir),
            "--slots",
            "--no-webui",
            "--no-context-shift",
            "--jinja",
            "--spec-type",
            "none",
            "--device",
            {"metal": "MTL0", "cuda": "CUDA0", "cpu": "none"}[self.backend],
        ]
        if self.backend == "cpu":
            args.extend(["--gpu-layers", "0"])
        return args

    def _process_reusable(self) -> bool:
        # Tests may inject an HTTP transport without an owned subprocess.
        return self._process is None or self._process.returncode is None

    async def start(self) -> None:
        if self._client and self._process_reusable():
            return
        if self._client:
            await self.close()
        from ..catalog import ensure_model, get_model
        from ..resources import check_admission

        started = time.perf_counter()
        model = get_model(self.profile.model_id)
        self._admission = await asyncio.to_thread(
            check_admission, self.settings, self.profile, model, self.backend
        )
        if not self._admission.allowed:
            raise AdmissionError("Memory admission refused: " + "; ".join(self._admission.reasons))
        binary = await asyncio.to_thread(ensure_engine, self.settings, self.backend)
        model_path = await asyncio.to_thread(ensure_model, self.settings, self.profile.model_id)
        self.fingerprint = RuntimeFingerprint(
            engine_revision=self.settings.engine_revision,
            model_sha256=model.sha256,
            cache_type_k=self.profile.cache_type,
            cache_type_v=self.profile.cache_type,
            context_size=self.profile.context_size,
        )
        self._slots_dir.mkdir(parents=True, exist_ok=True)
        self.settings.work_dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.settings.work_dir / f"llama-{self.backend}.log").open("ab")
        try:
            # Never reuse or terminate a server belonging to another process.
            import socket

            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", self.settings.engine_port))
            self._process = await asyncio.create_subprocess_exec(
                *self._arguments(binary, model_path),
                stdout=self._log,
                stderr=self._log,
            )
            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self.settings.engine_port}",
                timeout=self.settings.request_timeout,
                trust_env=False,
            )
            deadline = time.monotonic() + self.settings.startup_timeout
            while time.monotonic() < deadline:
                if self._process.returncode is not None:
                    raise RuntimeError(f"llama-server exited ({self._process.returncode}); see log")
                try:
                    response = await self._client.get("/health", timeout=1)
                    if response.status_code == 200:
                        await self._capture_native_props()
                        self.startup_ms = elapsed_ms(started)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
            raise TimeoutError("llama-server startup timed out")
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        await self.cancel()
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), 10)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None
        self._native_props = None
        self._effective_settings = None
        if self._log:
            self._log.close()
            self._log = None

    async def configure(self, profile: RuntimeProfile) -> None:
        if self._lock.locked():
            raise RuntimeError("Runtime has an active request")
        async with self._lock:
            if profile == self.profile and self._client and self._process_reusable():
                return
            previous = self.profile
            was_started = self._client is not None and self._process_reusable()
            await self.close()
            self.profile = profile
            try:
                await self.start()
            except Exception as rejected:
                self.profile = previous
                if was_started:
                    try:
                        await self.start()
                    except Exception as rollback:
                        raise RuntimeError(
                            f"Configuration failed ({rejected}); prior runtime restart also failed"
                        ) from rollback
                raise

    async def _capture_native_props(self) -> None:
        response = await self._http().get("/props")
        response.raise_for_status()
        props = response.json()
        defaults = props.get("default_generation_settings", {})
        actual_context = defaults.get("n_ctx")
        slots = props.get("total_slots")
        if slots is not None and slots != 1:
            raise RuntimeError(f"Native engine exposes {slots} slots; exactly one is required")
        if isinstance(actual_context, int) and actual_context < self.profile.context_size:
            raise RuntimeError(
                f"Native context {actual_context} is smaller than requested "
                f"{self.profile.context_size}"
            )
        self._native_props = {
            key: props[key]
            for key in (
                "default_generation_settings",
                "total_slots",
                "model_alias",
                "model_ftype",
                "model_path",
                "modalities",
                "build_info",
                "endpoint_slots",
            )
            if key in props
        }
        # This pinned API reports context and slots, but omits batch/cache types.
        # Preserve unknown values rather than relabeling requested flags as measurements.
        self._effective_settings = {
            "observed_from": "native /props",
            "effective_n_ctx": actual_context,
            "total_slots": slots,
            "effective_n_batch": props.get("n_batch", defaults.get("n_batch")),
            "effective_n_ubatch": props.get("n_ubatch", defaults.get("n_ubatch")),
            "cache_type_k": props.get("cache_type_k", defaults.get("cache_type_k")),
            "cache_type_v": props.get("cache_type_v", defaults.get("cache_type_v")),
            "build_info": props.get("build_info"),
        }

    async def _snapshot(self) -> dict:
        from ..resources import capture_resources

        snapshot = await asyncio.to_thread(capture_resources, self.settings.cache_dir)
        data = snapshot.model_dump(mode="json")
        native_rss = 0
        if self._process and self._process.returncode is None:
            try:
                native_rss = psutil.Process(self._process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        data["native_process_rss_bytes"] = native_rss
        return data

    async def health(self) -> dict:
        native = None
        if self._client and self._process_reusable():
            try:
                response = await self._client.get("/health", timeout=2)
                response.raise_for_status()
                native = response.json()
            except httpx.HTTPError:
                pass
        return {
            "status": "ready" if native else "stopped",
            "backend": self.backend,
            "profile": self.profile.model_dump(),
            "fingerprint": self.fingerprint.model_dump() if self.fingerprint else None,
            "engine_revision": self.settings.engine_revision,
            "startup_ms": self.startup_ms,
            "native_exit_code": self._process.returncode if self._process else None,
            "native_props": self._native_props,
            "effective_settings": self._effective_settings,
            "admission": self._admission.model_dump(mode="json") if self._admission else None,
            "native": native,
            "resources": await self._snapshot(),
        }

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Runtime has not been started")
        if not self._process_reusable():
            raise RuntimeError("Owned native llama-server exited; configure runtime to restart")
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        response = await self._http().post(path, json=payload)
        response.raise_for_status()
        value = response.json()
        if "error" in value:
            raise RuntimeError(f"Native server error: {value['error']}")
        return value

    async def format_chat(self, messages: list[dict]) -> str:
        result = await self._post("/apply-template", {"messages": messages})
        prompt = result.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("Native chat template did not return a prompt string")
        return prompt

    async def tokenize(self, prompt: str) -> list[int]:
        result = await self._post("/tokenize", {"content": prompt, "add_special": True})
        tokens = result["tokens"]
        if not all(type(token) is int for token in tokens):
            raise RuntimeError("Tokenizer returned non-integer tokens")
        return tokens

    async def _erase(self) -> None:
        if not self._process_reusable():
            return  # Process exit already released its native slot.
        await self._post("/slots/0?action=erase", {})

    async def cancel(self) -> None:
        active = self._active
        if active and active is not asyncio.current_task() and not active.done():
            active.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active

    async def _sample_peak(self) -> None:
        snapshot = await self._snapshot()
        self._peak_memory["sample_count"] += 1
        for source, target in [
            ("process_rss_bytes", "process_rss_max_bytes"),
            ("native_process_rss_bytes", "native_process_rss_max_bytes"),
            ("swap_used_bytes", "swap_used_max_bytes"),
            ("gpu_used_bytes", "gpu_used_max_bytes"),
        ]:
            value = snapshot.get(source)
            if value is not None:
                self._peak_memory[target] = max(value, self._peak_memory.get(target, value))
        for source, target in [
            ("ram_available_bytes", "ram_available_min_bytes"),
            ("gpu_free_bytes", "gpu_free_min_bytes"),
        ]:
            value = snapshot.get(source)
            if value is not None:
                self._peak_memory[target] = min(value, self._peak_memory.get(target, value))

    async def _monitor_memory(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            try:
                await self._sample_peak()
            except Exception as error:
                self._peak_memory["sampling_error"] = str(error)
                return

    @contextlib.asynccontextmanager
    async def _slot(self):
        if self._lock.locked():
            raise RuntimeError("Runtime supports one active request")
        async with self._lock:
            self._active = asyncio.current_task()
            self._peak_memory = {"sample_period_ms": 100, "sample_count": 0}
            monitor = None
            failure = None
            try:
                await self._sample_peak()
                monitor = asyncio.create_task(self._monitor_memory())
                yield
            except BaseException as error:
                failure = error
                raise
            finally:
                # Close native streams before erase so cancellation releases the slot.
                with anyio.CancelScope(shield=True):
                    try:
                        with anyio.fail_after(min(10, self.settings.request_timeout)):
                            await self._erase()
                    except Exception as cleanup_error:
                        if failure is None:
                            raise
                        failure.add_note(
                            f"Native slot cleanup failed: {type(cleanup_error).__name__}: "
                            f"{cleanup_error}"
                        )
                    finally:
                        if monitor:
                            monitor.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await monitor
                        self._active = None

    def _check_context(self, tokens: list[int], output: int = 1) -> None:
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Prompt tokens must be nonnegative integers")
        if len(tokens) + output > self.profile.context_size:
            raise ValueError("Prompt plus output exceeds configured context")

    async def prefill(self, token_ids: list[int]) -> PrefillArtifact:
        if self.fingerprint is None:
            raise RuntimeError("Runtime has not been started")
        self._check_context(token_ids)
        if len(token_ids) < 2:
            raise ValueError("Prefill requires at least two canonical prompt tokens")
        if self._admission is not None:
            await asyncio.to_thread(
                ensure_state_disk_space,
                self.settings.work_dir,
                self._admission.budgets["state_bytes"],
                self.settings.disk_reserve_bytes,
            )
        async with self._slot():
            await self._erase()
            start = time.perf_counter()
            completion = await self._post(
                "/completion",
                {
                    "prompt": token_ids[:-1],
                    "n_predict": 1,
                    "cache_prompt": False,
                    "id_slot": 0,
                    "temperature": 0,
                    "seed": 0,
                    "stream": False,
                    "return_tokens": True,
                    "ignore_eos": True,
                },
            )
            prefill_ms = elapsed_ms(start)
            timings = completion.get("timings", {})
            if timings.get("prompt_n") != len(token_ids) - 1 or timings.get("cache_n") != 0:
                raise RuntimeError("Native prefill did not evaluate the complete N-1 prefix")
            artifact_id = uuid4().hex
            filename = f"{artifact_id}.bin"
            temporary = self._slots_dir / filename
            directory = self.settings.artifacts_dir / artifact_id
            try:
                export_start = time.perf_counter()
                saved = await self._post("/slots/0?action=save", {"filename": filename})
                if saved.get("n_saved") != len(token_ids) - 1:
                    raise RuntimeError("Native slot saved wrong token boundary; require N-1")
                directory.mkdir(parents=True)
                state = directory / "state.bin"
                await asyncio.to_thread(shutil.move, temporary, state)
                digest = await asyncio.to_thread(sha256_file, state)
                await self._sample_peak()
                artifact = PrefillArtifact(
                    manifest=ArtifactManifest(
                        artifact_id=artifact_id,
                        model_id=self.profile.model_id,
                        fingerprint=self.fingerprint,
                        token_ids=token_ids,
                        prefix_tokens=len(token_ids) - 1,
                        state_bytes=state.stat().st_size,
                        state_sha256=digest,
                        source_backend=self.backend,
                        expires_at=datetime.now(UTC)
                        + timedelta(seconds=self.settings.artifact_ttl_seconds),
                        timings={
                            "prefill_ms": prefill_ms,
                            "export_ms": elapsed_ms(export_start),
                            "native_prefill_ms": timings.get("prompt_ms", 0),
                            "prefill_tps": timings.get("prompt_per_second", 0),
                            "native_save_ms": saved.get("timings", {}).get("save_ms", 0),
                            **{
                                export: self._peak_memory[source]
                                for source, export in [
                                    ("process_rss_max_bytes", "peak_process_rss_bytes"),
                                    ("gpu_used_max_bytes", "peak_gpu_used_bytes"),
                                ]
                                if source in self._peak_memory
                            },
                            "memory_sample_count": self._peak_memory["sample_count"],
                            "memory_sample_period_ms": self._peak_memory["sample_period_ms"],
                        },
                    ),
                    state_path=state,
                )
                write_manifest(artifact)
                return artifact
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            finally:
                temporary.unlink(missing_ok=True)

    async def generate(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent]:
        if request.mode != "local":
            raise ValueError(
                "Native runtime only implements local generation; use Pipeline for split"
            )
        start = time.perf_counter()
        tokenize_start = time.perf_counter()
        tokens = request.token_ids
        if tokens is None:
            tokens = await self.tokenize(request.prompt)
        metrics = RunMetrics(
            tokenize_ms=elapsed_ms(tokenize_start) if request.token_ids is None else 0
        )
        self._check_context(tokens, request.max_tokens)
        metrics.memory["before"] = await self._snapshot()
        async with self._slot():
            await self._erase()
            async with contextlib.aclosing(
                self._completion(tokens, request, metrics, start, cached=False)
            ) as stream:
                async for event in stream:
                    yield event

    async def decode(
        self, artifact: PrefillArtifact, request: DecodeRequest
    ) -> AsyncGenerator[GenerationEvent]:
        start = time.perf_counter()
        await asyncio.to_thread(validate_artifact, artifact, self.fingerprint)
        if artifact.manifest.model_id != self.profile.model_id:
            raise ValueError("Artifact model does not match runtime")
        await asyncio.to_thread(
            ensure_state_disk_space,
            self.settings.work_dir,
            artifact.manifest.state_bytes,
            self.settings.disk_reserve_bytes,
            1,
        )
        tokens = artifact.manifest.token_ids
        self._check_context(tokens, request.max_tokens)
        m = artifact.manifest
        metrics = RunMetrics(
            prefill_ms=m.timings.get("prefill_ms", 0),
            export_ms=m.timings.get("export_ms", 0),
            prefill_tps=m.timings.get("prefill_tps"),
            state_bytes=m.state_bytes,
        )
        metrics.memory["before"] = await self._snapshot()
        async with self._slot():
            await self._erase()
            temporary = self._slots_dir / f"restore-{uuid4().hex}.bin"
            try:
                restore_start = time.perf_counter()
                await asyncio.to_thread(shutil.copyfile, artifact.state_path, temporary)
                restored = await self._post("/slots/0?action=restore", {"filename": temporary.name})
                if restored.get("n_restored") != m.prefix_tokens:
                    raise RuntimeError("Native slot restored wrong token boundary; require N-1")
                metrics.restore_ms = elapsed_ms(restore_start)
                yield GenerationEvent(kind="stage", stage="restore", elapsed_ms=metrics.restore_ms)
                async with contextlib.aclosing(
                    self._completion(tokens, request, metrics, start, cached=True)
                ) as stream:
                    async for event in stream:
                        yield event
            finally:
                temporary.unlink(missing_ok=True)

    async def _completion(
        self,
        tokens: list[int],
        request: DecodeRequest | GenerationRequest,
        metrics: RunMetrics,
        start: float,
        cached: bool,
    ) -> AsyncGenerator[GenerationEvent]:
        payload = {
            "prompt": tokens,
            "n_predict": request.max_tokens,
            "cache_prompt": cached,
            "id_slot": 0,
            "temperature": request.temperature,
            "seed": request.seed,
            "ignore_eos": request.ignore_eos,
            "stream": True,
            "return_tokens": True,
            "timings_per_token": True,
        }
        metrics.memory["peak"] = self._peak_memory
        text_parts: list[str] = []
        first_token: float | None = None
        last_token: float | None = None
        generated_tokens = 0
        final: dict | None = None
        yield GenerationEvent(kind="stage", stage="decode" if cached else "prefill")
        async with self._http().stream("POST", "/completion", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                item = json.loads(raw)
                if "error" in item:
                    raise RuntimeError(f"Native generation failed: {item['error']}")
                timing = item.get("timings", {})
                if (
                    cached
                    and timing
                    and (timing.get("cache_n") != len(tokens) - 1 or timing.get("prompt_n") != 1)
                ):
                    raise RuntimeError("State handoff failed: require cache_n=N-1 and prompt_n=1")
                chunk = item.get("content", "")
                ids = item.get("tokens", [])
                if chunk or ids:
                    now = time.perf_counter()
                    if first_token is None:
                        first_token = now
                        metrics.ttft_ms = (now - start) * 1000
                    last_token = now
                    generated_tokens += len(ids)
                    text_parts.append(chunk)
                    yield GenerationEvent(
                        kind="token",
                        text=chunk,
                        token_id=ids[0] if len(ids) == 1 else None,
                        elapsed_ms=elapsed_ms(start),
                        data={"token_ids": ids},
                    )
                if item.get("stop"):
                    final = item
        if final is None:
            raise RuntimeError("Native completion stream ended without a terminal response")
        timing = final.get("timings", {})
        if not timing or final.get("truncated"):
            raise RuntimeError("Native completion missing timings or exceeded context")
        metrics.cache_n = int(timing["cache_n"])
        metrics.prompt_n = int(timing["prompt_n"])
        if cached and (metrics.cache_n != len(tokens) - 1 or metrics.prompt_n != 1):
            raise RuntimeError("State handoff failed: prompt was silently recomputed")
        if not cached and (metrics.cache_n != 0 or metrics.prompt_n != len(tokens)):
            raise RuntimeError("Local completion unexpectedly reused prompt cache")
        if cached:
            metrics.final_prompt_ms = timing.get("prompt_ms", 0)
        else:
            metrics.prefill_ms = timing.get("prompt_ms", 0)
            metrics.prefill_tps = timing.get("prompt_per_second")
        metrics.decode_ms = timing.get("predicted_ms", 0)
        metrics.decode_tps = timing.get("predicted_per_second")
        received_tokens = generated_tokens
        generated_tokens = int(timing.get("predicted_n", received_tokens))
        if first_token and last_token and last_token > first_token and received_tokens > 1:
            metrics.client_decode_tps = (received_tokens - 1) / (last_token - first_token)
        metrics.memory["after"] = await self._snapshot()
        await self._sample_peak()
        metrics.total_ms = elapsed_ms(start)
        result = GenerationResult(
            mode="split" if cached else "local",
            model_id=self.profile.model_id,
            prompt_tokens=len(tokens),
            generated_tokens=generated_tokens,
            text="".join(text_parts),
            metrics=metrics,
            fingerprint=self.fingerprint,
            local_profile=self.profile,
            effective_settings={"local": self._effective_settings or {}},
        )
        yield GenerationEvent(kind="result", result=result, elapsed_ms=metrics.total_ms)
