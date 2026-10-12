# Actual long-context local/split comparison

`scripts/run_long_context_benchmark.py` reuses the normal paired benchmark,
pipeline, native state checks and JSON/CSV/HTML renderer. It runs on the Mac
against a separately started CUDA worker. Each measured pair uses the same
canonical token IDs, Q4_K_M weights, Q8 attention cache and 128 generated tokens.
The default five pairs alternate execution order. The existing 32-token warmup
is recorded separately; it is not a full-length warmup or a cold-disk test.

The available experiment profiles are:

| Model | Canonical prompt | Requested context | Mac microbatch | CUDA microbatch |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 65,535 | 65,663 | 2,048 | 2,048 |
| Qwen3.5-27B | 32,768 | 32,896 | 2,048 | 12,288 |

The 2026-10-12 paired run was narrowed at the user's request to **4B only,
four complete pairs**. The 27B command below remains a reproducible launch
recipe; it was not executed in this paired run. Earlier CUDA-only 27B
measurements are recorded separately in [the calibration guide](calibration.md).

Use one model at a time. Stop the demo's normal servers before starting the
experiment, and restore them afterwards. Do not terminate unrelated workloads.
The 4B CUDA profile uses full GPU weight placement. The 27B profile uses CPU
FFN64 and recurrent, attention and output projection weights, retaining upstream
operation offload. Both use four native CPU threads and logical batch equal to
the canonical prompt length. Native context rounding and effective settings
are recorded separately.

## Source and worker

Commit and push changes, fetch them on the GPU host, and create an exact-SHA
deployment worktree. When the engine, dependency lock and native helper are
unchanged, `deployment/Dockerfile.worker-update` can reuse the verified native
image. Inspect its immutable image ID first, pass the checked local tag as
`BASE_IMAGE`, the ID as `BASE_IMAGE_ID`, and the new source SHA as `GIT_REVISION`.
Use the main Dockerfile when engine or dependencies change. Record both image
IDs and Git revisions; an image tag alone is not an immutable identity.

Launch the worker image with the following entrypoint followed by worker options,
or invoke `scripts/serve_calibration_worker.py` from the committed checkout:

```bash
python -c 'from pd_disaggregation_demo.calibration_worker import main; main()'
```

`--context-size` must exactly match the Mac's requested context.
`--batch-size`, `--ubatch-size` and CPU placement remain independent on each host.
The worker refuses profile changes that would need another memory probe.

Apply the [calibration guide's](calibration.md) live RAM budget, finite Docker
memory limit, equal memory/swap limit, NVIDIA CDI device, four-CPU limit and
read-only model mount. Publish only `127.0.0.1:8090:8090`; the Mac connects through
SSH. Keep `PDD_REQUEST_TIMEOUT=3600`, `PDD_STARTUP_TIMEOUT=600` and worker
`--timeout 3600` explicit. The guarded worker retains 1 GiB GPU and 2 GiB host
reserves, counts clean mmap pages as reclaimable, and monitors its whole lifetime.
Its health response exposes admission, placement and sampled memory/IO evidence.

For 4B, append:

```bash
--model-id qwen3.5-4b --context-size 65663 --batch-size 65535 \
  --ubatch-size 2048 --cache-type q8_0 --cpu-ffn 0 --threads 4
```

For 27B, append:

```bash
--model-id qwen3.5-27b --context-size 32896 --batch-size 32768 \
  --ubatch-size 12288 --cache-type q8_0 --cpu-ffn 64 --threads 4 \
  --cpu-recurrent-projections --cpu-attention-projections --cpu-output-projection
```

These experiment choices are not guaranteed safe on another resource snapshot;
keep the startup probe and live watchdog enabled.

## Link measurement

Use a Git-deployed copy of the module on the remote host:

```bash
uv run python scripts/measure_link.py --host gpu_home \
  --bytes 67108864 --repeats 3 --timeout 120 \
  --remote-source /absolute/exact-sha-checkout/src/pd_disaggregation_demo/link_probe.py \
  --remote-revision FULL_COMMIT_SHA --output link-throughput.json
```

This alternates GPU-host-to-Mac downloads and Mac-to-GPU-host uploads using
bounded buffers, disabled compression and verified byte counts/checksums. Wall
rates include SSH connection setup and Python hashing. The separately recorded
SSH command latency is not network RTT. The actual state-transfer timer in each
split sample is the relevant application throughput for TTFT.

## Mac commands

The explicit Mac overcommit flag is authorized only for this experiment. It
skips local RAM/Metal working-set admission while retaining telemetry and disk
checks; it cannot weaken CUDA safeguards. Remove it for ordinary launches.

```bash
uv run python scripts/run_long_context_benchmark.py \
  --model qwen3.5-4b --tokens 65535 --ubatch 2048 --repeats 5 \
  --allow-metal-memory-overcommit --timeout 3600 \
  --work-dir .pd-demo/long-context/4b-mac \
  --output .pd-demo/long-context/4b-benchmark.json

# After restarting the remote worker with the matching 27B profile:
uv run python scripts/run_long_context_benchmark.py \
  --model qwen3.5-27b --tokens 32768 --ubatch 2048 --repeats 5 \
  --allow-metal-memory-overcommit --timeout 3600 \
  --work-dir .pd-demo/long-context/27b-mac \
  --output .pd-demo/long-context/27b-benchmark.json
```

The command owns its Metal process and SSH tunnel, saves partial reports after
progress and each sample, and closes its resources on completion or cancellation.
It does not stop the separately managed CUDA container. Preserve that container's
exit/OOM state, launch budget and telemetry before stopping/removing it.

Interpret native prefill/decode rates separately from complete client-observed
TTFT. Split TTFT includes CUDA prefill, export, verified file transfer, Metal
restore and final prompt evaluation. Every split sample must prove cache N−1 and
prompt evaluation 1; local samples must prove cache 0 and prompt evaluation N.
Failed handoff must never be replaced by local prefill. Any ideal 1 Gbit/s values
in the standard report remain modeled projections, not measured network speed.
