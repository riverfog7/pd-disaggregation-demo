# Opt-in 27B microbatch calibration

Use this harness to test a specific Qwen3.5-27B placement, prompt length and microbatch on one machine. It is a separate experiment: it does not weaken ordinary worker admission or change upstream inference kernels. Estimated admission is not a measured result, and the watchdog cannot guarantee that OOM will never occur.

The [calibrator](../src/pd_disaggregation_demo/calibration.py) uses Q8_0 attention cache, FlashAttention on, one native slot, no speculative decoding, mmap weights, no repacking and no CUDA pinned host-weight buffers. CUDA defaults to 64 CPU FFN layers; Metal defaults to zero. CPU placement names describe **weight storage**. Large supported operations can still be offloaded to CUDA. `--cpu-output-projection` can also move the final logit projection to CPU.

The optional `--cpu-recurrent-projections`, `--cpu-output-projection` and `--cpu-attention-projections` flags pass identical tensor overrides to the estimator and server. The attention flag selects only q/k/v/output projection weights; it does not move Q/K normalization, KV cache or recurrent state. Record every placement flag when comparing runs.

FlashAttention avoids the full attention-score matrix, but does not eliminate KV cache, recurrent state or activation/compute buffers. Increasing context or microbatch can still exhaust memory. The server-faithful `pd-server-fit-params` helper applies one-slot, non-speculative output limits before calling pinned upstream memory accounting. The generic `llama-fit-params` remains available, but reserves a different output graph and must not substitute for this probe. Runtime chunk shapes can resize scheduler buffers beyond the no-allocation estimate.

## Prerequisites and source provenance

Run the CUDA recipe **on the Linux GPU host**. It assumes Docker with cgroup v2, NVIDIA CDI device `nvidia.com/gpu=all`, host Python 3, `uv`, and an existing image built from this project's [Dockerfile](../Dockerfile). The image must contain Python dependencies, the pinned `llama-server` and matching `pd-server-fit-params` beside it under `/opt/llama/bin`. The helper's compiled C++ source hash must match the source mounted for the run. The harness refuses a missing/stale helper instead of falling back to generic estimates.

The pinned 27B GGUF must already exist at `cache/models/qwen3.5-27b/Qwen3.5-27B-Q4_K_M.gguf` under the chosen data directory. The harness checks size/hash once before the sweep and never downloads weights. Allow disk space for native state exports plus the existing 20 GiB reserve. Incomplete or missing models are rejected.

Transfer code through Git: commit and push on the source machine, fetch on the GPU host, then check out the **full commit SHA** in a clean checkout/worktree. Do not patch the deployed source with file-copy commands. Source, image-building commit and immutable image ID are separate evidence. Reusing an older compatible image is possible, but preserve its different build revision and let the probe verify library/source compatibility.

Only one calibration/native GPU worker may run at a time; GPU 0 must be free of competing inference work. Stop the demo's owned worker using its lifecycle command before measuring RAM or launching calibration; do not terminate unrelated processes. Do not compile images, run another inference job or alter placement during a timed run. The recipe limits the container to four CPU cores, matching the harness's default four native threads.

## RAM budget, file cache and swap

After stopping the worker, choose a finite cap in bytes:

```text
container memory = floor_256MiB(MemAvailable − 2GiB − 256MiB)
container combined memory+swap limit = the same value
```

The extra 256 MiB is a launch cushion. This is a bounded allowance, not a reservation of unused RAM. Recompute it before each experiment; the live host and GPU guards still apply. `MemAvailable` already includes reclaimable file cache, so adding the old cache size again double-counts capacity. The container cap includes anonymous/kernel memory and cache; it is not an allowance solely for model pages. See [Linux meminfo semantics](https://docs.kernel.org/filesystems/proc.html#meminfo).

Equal positive Docker `--memory` and `--memory-swap` limits prohibit **container** swap. The harness verifies effective `memory.swap.max=0` and a finite real cgroup limit. Host swap can remain enabled or contain pages from other processes; a host `SwapUsed` value is not evidence that this container swapped. See [Docker swap-limit semantics](https://docs.docker.com/engine/containers/resource_constraints/#--memory-swap-details).

The pinned mmap loader already uses shared read-only file mappings and Linux prefetch hints. Hot weight pages are served from RAM; missing pages fault from storage. Raising the cap gives the kernel more room for cache without adding a custom streamer or locking the whole model. A repeated sequential weight scan can keep rereading storage when its working set exceeds available cache. Larger microbatches reduce the number of weight passes; more cache does not automatically eliminate disk reads.

Cgroup `file` bytes include other cached files and are **not the exact resident model cache**. Pages can remain charged to the group that first instantiated them while other groups share them, including pages left by prior verification/loading. Keep the same model inode, preserve host/cgroup before/after snapshots, and compare warm measured repeats separately from startup. Do not globally drop caches to manufacture a cold run. [Kernel memory ownership](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-ownership) explains the charging behavior.

CUDA preflight reserves 1 GiB VRAM plus a 256 MiB driver cushion, 2 GiB host availability and space for nonreclaimable compute/context/staging. The live guard checks host availability, GPU free memory, nonreclaimable cgroup headroom and OOM counters throughout owned loading and requests. Clean cached file pages near `memory.max` alone do not trigger an abort. Pressure, native failure or a deadline stops escalation and closes only the owned native process. Metal retains its separate 6 GiB unified-memory reserve and working-set check.

## Reproducible CUDA recipe

The blocks below form one Bash session. They do not fetch weights or build an image. Set `DATA_ROOT` to the existing worker data directory and, if needed, `IMAGE_REF` to an already-built compatible image. The example compares 8,192 and 12,288 microbatches at 32,768 prompt tokens; it asserts no measured outcome. Use a fresh output directory for each run.

```bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_SHA="$(git rev-parse HEAD)"
test -z "$(git status --porcelain -- src scripts pyproject.toml uv.lock Dockerfile)"
IMAGE_REF="${IMAGE_REF:-pd-disaggregation-demo:${SOURCE_SHA}}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
IMAGE_SOURCE_SHA="$(docker image inspect --format \
  '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE_ID")"
IMAGE_SOURCE_SHA="$(git rev-parse "${IMAGE_SOURCE_SHA}^{commit}")"
DATA_ROOT="$(realpath "${DATA_ROOT:-$REPO_ROOT/.pd-worker}")"
PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
test -r "$DATA_ROOT/cache/models/qwen3.5-27b/Qwen3.5-27B-Q4_K_M.gguf"

# Run the lifecycle stop from the checkout/settings that owns the demo worker.
uv run pd-demo stop linux

RUN_ROOT="$REPO_ROOT/.pd-demo/calibration-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
SETTINGS_FILE="$RUN_ROOT/settings.env"
cat > "$SETTINGS_FILE" <<'ENV'
PDD_ENGINE_PORT=8181
PDD_STARTUP_TIMEOUT=600
PDD_REQUEST_TIMEOUT=1800
ENV

calibration_memory_budget() {
python3 - <<'PY'
from pathlib import Path
available = next(int(line.split()[1]) * 1024
                 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
quantum = 256 * 1024**2
budget = ((available - 2 * 1024**3 - quantum) // quantum) * quantum
if budget <= 0:
    raise SystemExit('No memory budget remains after host reserve and cushion')
print(budget)
PY
}
CALIBRATION_MEMORY_BYTES="$(calibration_memory_budget)"
printf 'Source: %s\nImage build: %s\nImage ID: %s\nRAM cap bytes: %s\n' \
  "$SOURCE_SHA" "$IMAGE_SOURCE_SHA" "$IMAGE_ID" "$CALIBRATION_MEMORY_BYTES"
```

This helper mounts committed source, settings and existing weights read-only, with a writable directory for native temporary state and reports. It keeps container logs and actual exit/OOM status even when the saved report is partial. It overrides the worker image entrypoint and disables its API healthcheck; calibration has its own private native listener.

```bash
run_calibration() {
  local name="$1" destination="$2" code memory_bytes
  shift 2
  mkdir -p "$destination"
  memory_bytes="$(calibration_memory_budget)"
  printf '%s\n' "$memory_bytes" > "$destination/memory-cap-bytes.txt"
  cat /proc/meminfo > "$destination/host-meminfo-before.txt"
  cat /proc/vmstat > "$destination/host-vmstat-before.txt"
  if docker run --name "$name" --init --no-healthcheck --network none \
    --device nvidia.com/gpu=all --cpus 4 --pids-limit 256 \
    --memory "$memory_bytes" --memory-swap "$memory_bytes" \
    --security-opt no-new-privileges:true --cap-drop ALL \
    --user "$(id -u):$(id -g)" --workdir /app --entrypoint python \
    --mount "type=bind,src=$REPO_ROOT,dst=/source,readonly" \
    --mount "type=bind,src=$SETTINGS_FILE,dst=/app/.env,readonly" \
    --mount "type=bind,src=$DATA_ROOT/cache,dst=/data/cache,readonly" \
    --mount "type=bind,src=$destination,dst=/data/work" \
    -e PYTHONPATH=/source/src -e PDD_CACHE_DIR=/data/cache -e PDD_WORK_DIR=/data/work \
    -e PDD_ENGINE_BINARY=/opt/llama/bin/llama-server -e PDD_GIT_REVISION="$SOURCE_SHA" \
    -e PDD_CALIBRATION_SOURCE_REVISION="$SOURCE_SHA" -e PDD_CALIBRATION_IMAGE_ID="$IMAGE_ID" \
    "$IMAGE_ID" /source/scripts/calibrate_microbatch.py \
    --backend cuda --model-id qwen3.5-27b \
    --model-path /data/cache/models/qwen3.5-27b/Qwen3.5-27B-Q4_K_M.gguf \
    --prompt-tokens "$PROMPT_TOKENS" --output-tokens 128 --cpu-ffn 64 \
    --cpu-recurrent-projections --cpu-output-projection --cpu-attention-projections \
    --timeout 1800 --output /data/work/calibration.json "$@" \
    > "$destination/container.log" 2>&1; then code=0; else code=$?; fi
  if docker inspect "$name" > "$destination/container-inspect.json"; then
    python3 - "$destination" "$code" "$SOURCE_SHA" "$IMAGE_SOURCE_SHA" "$IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
state = json.loads((directory / 'container-inspect.json').read_text())[0]['State']
(directory / 'process-status.json').write_text(json.dumps({
    'container_exit_code': state['ExitCode'], 'oom_killed': state['OOMKilled'],
    'docker_cli_exit_code': int(sys.argv[2]), 'source_revision': sys.argv[3],
    'image_source_revision': sys.argv[4], 'image_id': sys.argv[5],
}, indent=2) + '\n')
PY
    docker rm "$name" > /dev/null
  fi
  cat /proc/meminfo > "$destination/host-meminfo-after.txt"
  cat /proc/vmstat > "$destination/host-vmstat-after.txt"
  return "$code"
}

# Metadata-only admission: no native server or generation is started.
run_calibration "pd-calibration-probe-$$" "$RUN_ROOT/probe" \
  --probe-only --ubatches 8192 12288

# Review estimated_safe_unmeasured versus estimated_skip in the probe JSON.
# Both candidates are checked again before load; no admission bypass is used.
# Each profile gets one full-prompt warmup and two measured repeats.
run_calibration "pd-calibration-measured-$$" "$RUN_ROOT/measured" \
  --ubatches 8192 12288 --repeats 2
```

An estimated candidate can still stop on live pressure or a changed resource snapshot. Keep the stopped report; do not bypass its guard or relabel it as safe. A later run can test a smaller candidate in a new directory. Candidates after a stop may not have been attempted. For more repetitions, change `--repeats 2` to `--repeats 3` and recompute the RAM cap before launch. To test a longer prompt, set `PROMPT_TOKENS` before the setup block and probe again; a 65,535-token prompt does not require increasing microbatch above 16,384. Every profile gets its own full-prompt warmup; candidate values are tried in ascending order. A finite deadline applies separately to loading, token preparation and each request. The native startup and HTTP settings in the mounted file can impose earlier deadlines.

SIGTERM/SIGINT allow owned-process cleanup and partial report persistence. If cancelling from another terminal, stop only the named calibration container and allow cleanup time; retain its logs and inspect state before removing it. An external kill/OOM can prevent Python's final report update, so the container sidecar is independent evidence.

For a Metal experiment on the Mac, use the same source and already-downloaded model with `uv run python scripts/calibrate_microbatch.py --backend metal --output PATH`; do not run it beside another Metal worker. Metal's preflight is conservative about summed host/device estimates and can skip all candidates. Skips are useful evidence, not permission to override the unified-memory reserve.

## Prompt boundaries and timings

`--prompt-tokens N` means **N canonical token IDs**, prepared with the benchmark corpus. The harness sets logical batch to N and requested context to at least `N + output_tokens`. For the example, N is 32,768 and requested context is 32,896; native context can round upward and is recorded separately. The supplied 8,192 and 12,288 microbatches are independent of prompt length. A separate 65,535-token prompt requests context 65,663 with the same 128-token output allowance; probe each changed context before measuring.

CUDA calls the existing strict prefill/export path: evaluate and save **N−1** tokens, sample/discard one token, then export native state. This example actually prefills/saves **32,767** tokens. A 12,288 microbatch exercises two full chunks and one 8,191-token remainder. With N=65,535, the prefix is 65,534; a 16,384 microbatch would exercise three full chunks and one 16,382-token remainder. Native prefix/cache/saved counts and the largest exercised chunk are checked/recorded; state files are deleted after each trial.

CUDA calibration is **prefill-only plus export**, not end-to-end TTFT, split transport or Metal decode. `native_prefill_ms` and `prefill_tps` come from the engine; client prefill, export and operation/trial wall timers cover different intervals. Metal calibration generates 128 local output tokens by default and records local TTFT/decode rates. Use the normal paired benchmark to measure local versus split TTFT.

Startup/load time is separate from warmup and measured requests. Model hashing, prior jobs, mmap prefetch and Linux file cache can warm disk pages before timed prefill. An empty **prompt/KV cache** does not mean a cold **disk/page cache**. These runs must not be described as cold-disk benchmarks without a separate controlled procedure and evidence.

## Render and interpret saved evidence

The [renderer](../src/pd_disaggregation_demo/calibration_report.py) performs no inference. Copy each run directory back with its JSON, `process-status.json` and adjacent `calibration.telemetry/` directory intact; telemetry paths can be resolved beside the report after relocation. Render on the source checkout:

```bash
uv run python scripts/render_calibration.py \
  "$RUN_ROOT/probe/calibration.json" "$RUN_ROOT/measured/calibration.json" \
  --output-prefix "$RUN_ROOT/comparison"
```

Outputs are `.summary.json`, `.summary.csv`, `.trials.csv` and standalone `.html`. The HTML embeds Plotly and separates measured trials, warmup/startup, estimated skips and failures. If a job failed, its available report can still be rendered by naming that JSON explicitly. Comparison medians use completed measured trials only. They do not establish a supported TTFT speedup or prove the largest possible batch.

GPU activity and disk timelines describe the **whole trial**, including export and client overhead, rather than isolated prefill kernels. NVML GPU/memory utilization, clocks, power and supported PCIe counters are sampled alongside native RSS, physical read bytes, major faults, main-process state and cgroup memory/IO pressure. Physical process reads include readahead; main-process state does not describe every worker thread. Low GPU utilization alone does not prove an SSD bottleneck. Correlate reads, faults and pressure with activity, and keep unavailable sensors explicit.

The watchdog requests roughly 100 ms sampling, records actual gaps, and can miss faster peaks. JSONL telemetry is bounded to 50,000 rows/16 MiB per candidate; summaries continue after truncation. A time-weighted GPU average or fraction below 20% describes sampled device activity, not a GPU-kernel duty-cycle trace. A clean Docker exit and unchanged OOM counters strengthen a completed result, but do not guarantee future runs will avoid OOM.
