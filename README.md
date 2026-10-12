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

The 4B profile is the normal application default and has passed CUDA-to-Metal validation on `gpu_home`. The 0.8B commands above provide a quicker correctness check. Admission reports required and available bytes; choosing a larger model does not override it. A running worker can switch only to models already fetched into its read-only model mount. See [deployment details](deployment/README.md) for asset prefetch, rollback, container budgets and troubleshooting.

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

TTFT uses the Mac monotonic clock through the first generated token, including handoff and final evaluation. Metadata-only chunks do not count. Engine prefill/decode rates are separate from client-observed decode rates. Split transfer does not inherently accelerate Mac decoding. A stage table compares tokenization, prefill, save, transfer, restore, final-token evaluation and unattributed overhead for the same local/split pair. The whole-request timeline marks the first token and then shows the remaining client elapsed time. Native decode timing is shown separately because it overlaps that interval. Graphs use vertical panels, actual prompt-size axis labels and wrapping legends for narrow windows. Transfer and memory views use the same responsive layout; full memory metric names remain in hover details.

The default sweep is 1k, 4k, 8k, 16k and 32k input tokens, 128 output tokens and five paired repeats, alternating local/split order. Enter 65536 in the dashboard to request 64k. Context is raised to cover prompt plus output; unsafe sizes are explicitly skipped. Canonical token preparation is recorded separately from paired TTFT, so both sides receive identical IDs. Weights are warmed, prompt caches are empty, and startup is reported separately. No cold-disk claim is made.

Speedup is the geometric mean of within-pair local/split TTFT ratios, with a seeded paired bootstrap 95% interval. A supported win requires its lower bound to exceed 1. Results also show losses. Five repeats do not establish reliable p95 latency. F16 and Q8 results are separate experiments. Sampled memory peaks include the interval and sample count and can miss shorter transients; unavailable GPU measurements remain unavailable.

Results also include an explicitly modeled **ideal 1Gbps** comparison: `measured split TTFT − measured transfer time + state bytes / 125,000,000 × 1000` milliseconds. All other observed overhead remains. This optimistic payload-only replacement omits framing and any hashing/disk work included in the replaced transfer timer; it is not a measured gigabit LAN result. Its paired ratio and confidence interval describe variability in the recorded samples under that assumption, not uncertainty in network performance. Modeled results never set the measured `supported_win` flag.

Reports are saved under `.pd-demo/reports/` as JSON, CSV and standalone HTML with embedded Plotly. The dashboard supports history comparison and downloads. Selecting saved runs also shows the most recent selected run’s exact stage table, whole-request timeline, and ideal 1Gbps projection. Keep large reports outside Git; `reports/` contains compact acceptance and benchmark evidence.

## Recorded 4B comparison

The [2026-10-12 rerun](reports/benchmark-4b-q8-rerun-2026-10-12.json) uses the same canonical prompts as the [original run](reports/benchmark-4b-q8-current-route.json): Qwen3.5-4B Q4_K_M, Q8 cache, 128 output tokens and five alternating pairs per size. All 30 requests passed native cache-boundary and token-count checks. Times below are medians; ratios are paired geometric means and need not equal the ratio of those medians.

| Prompt tokens | Local TTFT | Current-route split TTFT | Ideal 1Gbps modeled TTFT | Modeled paired speedup (95% CI) |
|---:|---:|---:|---:|---:|
| 1,024 | 2.19 s | 7.54 s | 1.86 s | 1.15× (1.06–1.24×) |
| 4,096 | 9.11 s | 16.54 s | 4.15 s | 2.21× (2.17–2.26×) |
| 8,192 | 18.97 s | 31.71 s | 7.31 s | 2.48× (2.30–2.67×) |

The current-route split loses at all three sizes, with transfer throughput medians of 91, 74 and 60 Mbit/s. At 8k, prefill reached 433 tokens/s on Metal and 1,985 tokens/s on CUDA; decode stayed on Metal at 29.97 tokens/s locally and 29.40 tokens/s after handoff. The ideal 1Gbps column is conditional on payload throughput and does not establish a measured gigabit LAN win. Compare complete intervals and per-trial timings in the reports; current SSH throughput and Mac processing rates changed between runs, so the comparison does not isolate the transport fix's effect.

The History tab can compare the rerun (`db03c2be`) with the original (`33c47212`) without rerunning inference. Full JSON, CSV and standalone HTML are under `.pd-demo/reports/`. Contexts were 8,192 (1k/4k) and 8,320 requested / 8,448 effective (8k); batch 512, microbatch 128, four CPU threads. Larger sizes are available through admission but were not measured in these runs.

An [interrupted attempt](reports/transport-rerun-failure.json) is preserved separately: an idle HTTP connection closed during 4k warmup. Remote control requests now use fresh connections, with no mutation retries. The successful rerun uses that fix for every pair.

## Recorded 27B calibration

The [2026-10-12 calibration evidence](reports/calibration-27b-2026-10-12.json) records real Qwen3.5-27B Q4_K_M prefill and native state export on the RTX 2080. FlashAttention and Q8 cache were enabled. Every request began with an empty prompt cache and evaluated/saved exactly **N−1** tokens. These are native prefill timings; 27B Metal decode and end-to-end split TTFT have not been measured.

| Canonical prompt | Microbatch | Measured repeats | Prefill median | Tokens/s median | Physical reads median | GPU activity median |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 2,048 | 1 | 99.14 s | 82.6 | 50.86 GB | 22.6% |
| 8,192 | 3,840 | 1 | 87.94 s | 93.1 | 37.53 GB | 21.6% |
| 8,192 | 8,192 | 3 | 32.32 s | 253.4 | 11.57 GB | 49.8% |
| 16,384 | 16,384 | 3 | 58.46 s | 280.2 | 14.00 GB | 59.3% |
| 32,768 | 8,192 | 2 | 165.88 s | 197.8 | 54.75 GB | 46.1% |
| 32,768 | 12,288 | 2 | 146.34 s | 224.0 | 41.94 GB | 52.4% |

The two **32k** configurations share the same prompt IDs, image/source revision, 10.75 GiB Docker RAM cap, four CPU threads and CPU weight placement. Increasing microbatch from 8,192 to 12,288 improved median throughput by **13.2%**, reduced prefill time by **11.8%**, and reduced physical reads by **23.4%** in this run. It changes four native prompt chunks into three. The 12,288 case retained at least **1.58 GiB GPU free memory** and **9.93 GiB host MemAvailable**; both candidates completed with zero cgroup OOM events and a clean process exit. Repetitions were sequential, not randomized; two samples per candidate do not establish a confidence interval or a universal optimum.

GPU activity, disk reads and pressure cover the **whole prefill/export trial**, not isolated kernels. Median full I/O stall time fell from 31.1 to 25.3 seconds, but GPU activity remained around 52%. Storage waits remain significant; these observations do not attribute every idle interval to storage rather than CPU, PCIe or synchronization.

The 8k rows use CPU FFN and recurrent-projection weights with an 8 GiB cap. The 16k/32k rows additionally put attention/output projection weights on CPU, using 9.25/10.75 GiB caps. Do not treat comparisons across those prompt lengths as changes to microbatch alone. The 16k/16,384 run also completed cleanly, with at least 1.07 GiB GPU free. A 65,535-token probe rejected the tested 10,240–16,384 microbatches under the VRAM reserve; its 8,192 warmup was intentionally cancelled when the experiment switched to 32k, so no completed 65k timing is claimed.

Linux mmap already supplies RAM file caching. In a separate identical 8k-profile comparison, increasing the container cap from 8 to 9.25 GiB increased sampled charged file-cache peaks from about 7.34 to 8.59 GiB, but prefill medians were 32.32 and 32.68 seconds: no observed gain in that comparison. At 32k the sampled charged file cache reached 9.75–9.94 GiB. These counters include other files and do not measure a permanently retained weight subset; sequential scans can evict and reread weights. Container swap was disabled; host swap activity is recorded separately.

The earlier 8k/8,192 runs completed their inference trials but exited 139 during GPU telemetry cleanup. Commit `5c5bb4c` fixes the sampler/NVML lifetime race; the later 16k and 32k runs exited 0. An earlier 3,072 candidate failed before inference on TCP port reuse, also fixed. Raw status and failures remain in the evidence rather than being counted as OOMs or hidden. The [calibration guide](docs/calibration.md) gives the guarded Docker recipe, timing definitions and report command. Full interactive graphs and CSVs are saved locally under `.pd-demo/reports/calibration-27b*`.

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

The 27B profile exposes existing `--cpu-ffn` placement in Linux launch/deploy and conservative microbatch controls. The opt-in calibration above validates specific CUDA prefill/export configurations on this 8 GiB GPU/16 GiB host. Ordinary worker admission remains conservative and can still refuse 27B; do not bypass it. Weight rereads can occur per microbatch, not once per whole prompt. Mac 27B admission was refused under the required 6 GiB reserve at the available-memory snapshot used for this experiment.

For the separate opt-in 27B experiment, see the [guarded calibration and report workflow](docs/calibration.md).

Settings use the `PDD_` environment prefix or `.env`. Useful values are `PDD_REMOTE_URL`, `PDD_REMOTE_SSH_HOST`, `PDD_LAN_SSH_HOST`, `PDD_REMOTE_REPO`, `PDD_PORT`, `PDD_ENGINE_PORT`, `PDD_CACHE_DIR`, and `PDD_WORK_DIR`. Cache defaults to `~/.cache/pd-disaggregation-demo`; run state defaults to `.pd-demo`. The native engine is pinned, has one slot, and disables auxiliary RAM prompt cache, checkpoints, speculative decoding and context shift.

## Validation

```bash
uv run --extra mac ruff check src tests scripts
uv run --extra mac ruff format --check src tests scripts
uv run --extra mac ty check src
uv run --extra mac pytest -q

# Opt-in real hardware, separate native port; also tests rejection and recovery.
uv run python scripts/validate_handoff.py --engine-port 8181
uv run python scripts/validate_handoff.py --prompt-tokens 8192 --engine-port 8181
uv run python scripts/validate_handoff.py --engine-port 8181 \
  --worker-url http://127.0.0.1:8090 --output .pd-demo/cuda-metal-acceptance.json
```

[Metal acceptance evidence](reports/metal-acceptance.json), [4B checks](reports/metal-4b-acceptance.json), and [8k-token checks](reports/metal-8k-acceptance.json) record actual F16 and Q8 same-prefix continuations and artifact rejection/recovery. [CUDA-to-Metal checks](reports/cuda-metal-acceptance.json) and [4B at 8k tokens](reports/cuda-metal-4b-8k-acceptance.json) passed on both cache types, including exact cache boundaries, repeated restoration and all five artifact rejection/recovery cases. The hardware script fails on inconsistent same-backend continuations; CUDA/Metal numerical differences remain recorded observations. Unit/integration tests additionally use fake native/worker servers for failures, cancellation and deployment command validation.

Benchmarks use the current SSH route and report its observed state-transfer throughput. The 1Gbps model lets you compare the same measurements under an ideal link assumption without presenting that assumption as a measured LAN speedup.

The [verification status](reports/verification-status.json) distinguishes executed checks from unvalidated configurations. As of 2026-10-12, 317 automated tests pass. Native CUDA/Metal handoff, successful paired browser execution, HTTP cancellation/disconnection, interrupted state transfer, worker restart recovery, unsafe-memory refusal and CLI Linux launch/stop have been exercised. The CUDA image builds with the pinned upstream linker flag and reuses a bounded persistent compilation cache. The 9B profile remains untested on hardware; 27B CUDA prefill/export is calibrated as described above, while its cross-machine TTFT remains unmeasured. The 1Gbps comparison remains a model.

## Layout and source workflow

One uv project contains `config/`, `models.py`, `catalog.py`, `runtime/`, `client/`, `backend/`, `deployment/`, `benchmarks/`, `ui/`, `api.py` and `cli.py` under `src/pd_disaggregation_demo`. The Linux API is a package module; there is no copied `server/` project. FastAPI, HTTPX, Typer, Gradio, Plotly, Hugging Face Hub, psutil and NVML provide the orchestration pieces.

Work is committed atomically on `codex/pd-demo`. Deployment pushes a committed SHA, fetches it remotely, creates an exact-SHA worktree, builds a SHA-tagged image and verifies health metadata. The previous healthy deployment is retained for rollback. Weights, native binaries, build caches and state files stay outside Git. [macuda's router](https://github.com/Davinchy/macuda/blob/8b2a17d2025795def022d75a85f05c623ad673e1/tools/disagg-router.py) informed the native slot handoff; no macuda driver is required.
