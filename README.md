# Prefill / decode disaggregation demo

Compare **Mac prefill → Mac decode** with **Linux CUDA prefill → native state transfer → Mac decode**. The main dashboard measures TTFT, prefill/decode throughput, transfer cost, and sampled memory use. Chat is a secondary view of the same pipeline.

The inference engine is unmodified [llama.cpp at `23b0202a189c44a54625aadcb37a946dd1d6278d`](https://github.com/ggml-org/llama.cpp/tree/23b0202a189c44a54625aadcb37a946dd1d6278d). Both machines use byte-identical pinned Q4_K_M GGUF files. Native slot serialization transfers attention **and recurrent** state for Qwen3.5; there is no cache tensor conversion or custom inference kernel.

## Run the demo

Requirements: `uv`, Python 3.13 (uv can install it), Git, CMake and Xcode command-line tools on the Mac; Docker Compose, Buildx and NVIDIA CDI on Linux. Source deployment uses Git over SSH. The Linux inference runtime lives in Docker.

```bash
uv sync --extra mac
uv run pd-demo setup --model qwen3.5-0.8b
uv run pd-demo doctor --remote --model qwen3.5-0.8b

# Commit every source change before deployment. A full SHA is required.
uv run pd-demo deploy --host gpu_home --ref "$(git rev-parse HEAD)" \
  --model qwen3.5-0.8b --context 2048

# Native Metal engine, local API, SSH tunnel, and web UI.
uv run --extra mac pd-demo serve mac --model qwen3.5-0.8b --context 2048
```

Open **http://127.0.0.1:8080/demo/**. The API documentation is at **http://127.0.0.1:8080/docs**. `serve mac --detach`, `status mac`, and `stop mac` manage a background Mac service. `--no-tunnel` leaves tunnel management to you. `--lan-host gpu_home_lan` chooses a separate SSH route for measurements without changing the deployment host.

`pd-demo status remote` inspects the deployed Docker worker. `pd-demo stop remote` stops only that worker while retaining its release and data; redeploy its committed SHA to start it again.

On a Linux checkout, the equivalent Docker launch is:

```bash
uv run pd-demo serve linux --detach --model qwen3.5-0.8b --context 2048
uv run pd-demo status linux
uv run pd-demo stop linux
```

The 4B profile is the normal application default. The 0.8B commands above fit the currently constrained `gpu_home` more easily. Admission reports required and available bytes; choosing a larger model does not override it. A running worker can switch only to models already fetched into its read-only model mount. See [deployment details](deployment/README.md) for asset prefetch, rollback, container budgets and troubleshooting.

## Commands and separate operations

`pd-demo --help` lists setup, doctor, model fetch/list, server lifecycle, deployment, rollback, prefill, decode, generate and benchmark commands.

The following standalone commands own a temporary native engine. Stop the Mac service first, or use their `--api-url http://127.0.0.1:8080` option to reuse it. Standalone split commands expect an existing SSH forward to port 8090.

```bash
uv run pd-demo generate --model qwen3.5-0.8b --prompt "Explain prefill." --max-tokens 64
uv run pd-demo prefill --model qwen3.5-0.8b --prompt-file prompt.txt
uv run pd-demo prefill --model qwen3.5-0.8b --prompt-file prompt.txt --remote
# The prefill result prints the saved directory and manifest.
uv run pd-demo decode .pd-demo/artifacts/ARTIFACT_ID --max-tokens 128

uv run pd-demo benchmark --model qwen3.5-0.8b --sizes 1024,4096 \
  --repeats 5 --output-tokens 128 --api-url http://127.0.0.1:8080
```

API-prefill returns an artifact ID and manifest on the server. Native prefills save `manifest.json` and `state.bin` under `.pd-demo/artifacts/`. Artifacts expire after one hour by default. Decode requires a matching model hash, engine revision, cache precision, context and state format. Failed split requests remain failures: they never fall back to local prefill.

The HTTP API includes `/health`, `/capabilities`, `/configure`, `/prefill`, `/decode`, `/generate`, `/cancel`, `/artifacts/{id}/manifest`, `/artifacts/{id}/state`, `/benchmarks`, `/benchmarks/{job}/events`, `/reports/{run}/{json,csv,html}`, and text-only `/v1/chat/completions` (streaming and non-streaming). Streaming `/generate` and `/decode` use named SSE events with the same typed Python results. Benchmark submission returns a job ID; progress streams and partial reports survive failed/cancelled runs. There is one active request/session at a time.

## Python API

```python
import asyncio
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import GenerationRequest, RuntimeProfile
from pd_disaggregation_demo.pipeline import Pipeline


async def main():
    pipeline = Pipeline(Settings(), RuntimeProfile(model_id="qwen3.5-0.8b"))
    try:
        await pipeline.start()
        async for event in pipeline.generate(
            GenerationRequest(
                prompt="Explain the difference between prefill and decode.",
                mode="split",
                max_tokens=128,
            )
        ):
            if event.kind == "token":
                print(event.text, end="", flush=True)
            elif event.result:
                print(event.result.metrics.model_dump())
    finally:
        await pipeline.close()


asyncio.run(main())
```

`Pipeline.prefill(PrefillRequest(...))` and `Pipeline.decode(artifact, DecodeRequest(...))` expose the separate operations. `Pipeline.session()` reserves the complete sequence for the current task. A session rejects competing operations, including configuration changes.

## What is measured

The Mac tokenizes once. Remote prefill evaluates the first **N−1 tokens**, samples and discards one token, then saves native state. The client streams and hashes the state file, restores it on Metal, evaluates the final prompt token and starts decoding. Both the native saved/restored count and completion timings must prove **`cache_n=N−1`, `prompt_n=1`**. A silent full re-prefill is an error.

TTFT uses the Mac monotonic clock through the first generated token, including handoff and final evaluation. Metadata-only chunks do not count. Engine prefill/decode rates are separate from client-observed decode rates. Split transfer does not inherently accelerate Mac decoding. The timeline shows measured components; protocol overhead can leave a residual relative to total TTFT.

The default sweep is 1k, 4k, 8k, 16k and 32k input tokens, 128 output tokens and five paired repeats, alternating local/split order. Enter 65536 in the dashboard to request 64k. Context is raised to cover prompt plus output; unsafe sizes are explicitly skipped. Canonical token preparation is recorded separately from paired TTFT, so both sides receive identical IDs. Weights are warmed, prompt caches are empty, and startup is reported separately. No cold-disk claim is made.

Speedup is the geometric mean of within-pair local/split TTFT ratios, with a seeded paired bootstrap 95% interval. A supported win requires its lower bound to exceed 1. Results also show losses. Five repeats do not establish reliable p95 latency. F16 and Q8 results are separate experiments. Sampled memory peaks include the interval and sample count and can miss shorter transients; unavailable GPU measurements remain unavailable.

Reports are saved under `.pd-demo/reports/` as JSON, CSV and standalone HTML with embedded Plotly. The dashboard supports history comparison and downloads. Keep large reports outside Git; `reports/` contains compact acceptance evidence only.

## Models, memory and configuration

| Profile | Pinned GGUF size | Purpose |
|---|---:|---|
| `qwen3.5-0.8b` | 532,517,120 bytes | Correctness and transport checks |
| `qwen3.5-4b` | 2,740,937,888 bytes | Default performance comparison |
| `qwen3.5-9b` | 5,680,522,464 bytes | Larger comparison when admitted |
| `qwen3.5-27b` | 16,740,812,704 bytes | Constrained CPU-FFN/offload experiment |

[The catalog](src/pd_disaggregation_demo/catalog.py) pins each HF repository revision, filename and SHA256. Both machines verify size and hash before inference. Default cache precision is F16; Q8_0 is explicit. Recurrent state keeps the engine's native representation.

Downloads use a single HTTP stream with bounded chunks, resumable partial files and an atomic publish after checksum verification. Xet reconstruction is disabled to keep download RAM predictable on the remote host.

Admission reserves at least 1 GiB VRAM, 2 GiB Linux host RAM, 6 GiB Mac unified RAM and 20 GiB disk beyond estimated application needs. Weights, KV/recurrent state, compute and export/import buffers count toward the estimate. The Linux worker has equal Docker memory/swap limits and a watchdog that stops **only its own engine** after sustained reserve violations. `/configure` can recover it after pressure is resolved. It never kills unrelated processes.

The 27B profile exposes existing `--cpu-ffn` placement in Linux launch/deploy and conservative microbatch controls. Its weight distribution has not been calibrated on this 8 GiB GPU/16 GiB host, so admission can refuse it; it is not a claim of a validated 27B streaming configuration. Do not bypass admission to force it. Weight rereads can occur per microbatch, not once per whole prompt.

Settings use the `PDD_` environment prefix or `.env`. Useful values are `PDD_REMOTE_URL`, `PDD_REMOTE_SSH_HOST`, `PDD_LAN_SSH_HOST`, `PDD_REMOTE_REPO`, `PDD_PORT`, `PDD_ENGINE_PORT`, `PDD_CACHE_DIR`, and `PDD_WORK_DIR`. Cache defaults to `~/.cache/pd-disaggregation-demo`; run state defaults to `.pd-demo`. The native engine is pinned, has one slot, and disables auxiliary RAM prompt cache, checkpoints, speculative decoding and context shift.

## Validation

```bash
uv run --extra mac ruff check src tests scripts
uv run --extra mac ruff format --check src tests scripts
uv run --extra mac ty check src
uv run --extra mac pytest -q

# Opt-in real hardware, separate native port; also tests rejection and recovery.
uv run python scripts/validate_handoff.py --engine-port 8181
uv run python scripts/validate_handoff.py --engine-port 8181 \
  --worker-url http://127.0.0.1:8090 --output .pd-demo/cuda-metal-acceptance.json
```

[Metal acceptance evidence](reports/metal-acceptance.json) records actual F16 and Q8 same-prefix continuations and artifact rejection/recovery, rather than mocked timings. Unit/integration tests additionally use fake native/worker servers for failures, cancellation and deployment command validation. Cross-machine and browser verification results are recorded separately as they complete.

Performance acceptance needs the direct gigabit LAN route. The existing Internet SSH route measured around 2.2 MB/s during planning and is useful for functionality checks, not for claiming the intended LAN speedup.

## Layout and source workflow

One uv project contains `config/`, `models.py`, `catalog.py`, `runtime/`, `client/`, `backend/`, `deployment/`, `benchmarks/`, `ui/`, `api.py` and `cli.py` under `src/pd_disaggregation_demo`. The Linux API is a package module; there is no copied `server/` project. FastAPI, HTTPX, Typer, Gradio, Plotly, Hugging Face Hub, psutil and NVML provide the orchestration pieces.

Work is committed atomically on `codex/pd-demo`. Deployment pushes a committed SHA, fetches it remotely, creates an exact-SHA worktree, builds a SHA-tagged image and verifies health metadata. The previous healthy deployment is retained for rollback. Weights, native binaries, build caches and state files stay outside Git. [macuda's router](https://github.com/Davinchy/macuda/blob/8b2a17d2025795def022d75a85f05c623ad673e1/tools/disagg-router.py) informed the native slot handoff; no macuda driver is required.
