# Measurement details

[Back to the performance overview](../README.md#performance-and-progress)

Hardware measurements below were recorded on **2026-10-12**. They retain the observed wins, losses, settings, calibration failures and unmeasured cases.

## Recorded 4B 65k comparison

The [65,535-token experiment](../reports/benchmark-4b-65k-2026-10-12.json) completed **four alternating local/split pairs** with Q4_K_M weights, Q8 cache, FlashAttention and 128 generated tokens. Both engines used microbatch 2,048, logical batch 65,535 and four CPU threads; requested context was 65,663, rounded natively to 65,792. Every request passed its exact token/cache boundary checks, and all eight continuations matched in this fixture. The 32-token warmup was separate; this was not a full-length warmup or a cold-disk test.

| Metric (median) | Mac local | CUDA prefill → Mac decode |
|---|---:|---:|
| TTFT | 386.08 s | 151.12 s |
| Native prefill throughput | 169.8 tokens/s | 1,727.8 tokens/s |
| Native decode throughput | 19.08 tokens/s | 17.89 tokens/s |
| State transfer | — | 1.195 GB in 105.87 s |

The measured paired TTFT speedup was **2.522× (95% paired bootstrap interval 2.462–2.578×)** on the existing `gpu_home` SSH route. This supports a win for this four-pair experiment, not a universal speedup or reliable tail-latency estimate. Split prefill took a median 37.96 seconds, state export 3.17 seconds, and restore 0.37 seconds. Transfer remained the largest cost, at a median 90.33 Mbit/s. Stage medians are descriptive; the UI's exact-pair table provides additive timing accounting. Decode still ran on the Mac and did not improve.

The separate [bidirectional link probe](../reports/link-throughput-2026-10-12.json) used three verified 64 MiB transfers each way: median **73.50 Mbit/s (9.19 MB/s) GPU host → Mac**, and **177.27 Mbit/s (22.16 MB/s) Mac → GPU host**. These wall rates include fresh SSH setup and hashing, not only physical network transfer. The actual larger state transfer amortized that setup. Replacing only the measured state-transfer time with ideal 1Gbps payload time gives a **modeled 54.78-second split TTFT and 6.972× paired speedup**; this is not a measured gigabit LAN result.

The guarded CUDA container had a 10.25 GiB RAM cap and no swap, with at least **3.04 GiB GPU free** and **11.54 GiB host MemAvailable** observed. No cgroup OOM or watchdog failure occurred. Mac admission was explicitly waived for this run. The user shortened the original five-pair run to four complete pairs; the runner's raw cancellation record and next-dispatch cancellation are retained separately, and no fifth-pair timing is included. The requested 27B paired run was not started after the user limited scope to 4B.

Use the [long-context guide](long-context-benchmarks.md) to reproduce the setup. The History tab lists this run as `a148fe49`; full JSON, CSV and standalone HTML, including the stage table and all eight timelines, are under `.pd-demo/reports/`. Normal 4B services were restored after measurement.

## Earlier 4B 1k–8k comparison

The [2026-10-12 rerun](../reports/benchmark-4b-q8-rerun-2026-10-12.json) uses the same canonical prompts as the [original run](../reports/benchmark-4b-q8-current-route.json): Qwen3.5-4B Q4_K_M, Q8 cache, 128 output tokens and five alternating pairs per size. All 30 requests passed native cache-boundary and token-count checks. Times below are medians; ratios are paired geometric means and need not equal the ratio of those medians.

| Prompt tokens | Local TTFT | Current-route split TTFT | Ideal 1Gbps modeled TTFT | Modeled paired speedup (95% CI) |
|---:|---:|---:|---:|---:|
| 1,024 | 2.19 s | 7.54 s | 1.86 s | 1.15× (1.06–1.24×) |
| 4,096 | 9.11 s | 16.54 s | 4.15 s | 2.21× (2.17–2.26×) |
| 8,192 | 18.97 s | 31.71 s | 7.31 s | 2.48× (2.30–2.67×) |

The current-route split loses at all three sizes, with transfer throughput medians of 91, 74 and 60 Mbit/s. At 8k, prefill reached 433 tokens/s on Metal and 1,985 tokens/s on CUDA; decode stayed on Metal at 29.97 tokens/s locally and 29.40 tokens/s after handoff. The ideal 1Gbps column is conditional on payload throughput and does not establish a measured gigabit LAN win. Compare complete intervals and per-trial timings in the reports; current SSH throughput and Mac processing rates changed between runs, so the comparison does not isolate the transport fix's effect.

The History tab can compare the rerun (`db03c2be`) with the original (`33c47212`) without rerunning inference. Full JSON, CSV and standalone HTML are under `.pd-demo/reports/`. Contexts were 8,192 (1k/4k) and 8,320 requested / 8,448 effective (8k); batch 512, microbatch 128, four CPU threads. Larger sizes are available through admission but were not measured in these runs.

An [interrupted attempt](../reports/transport-rerun-failure.json) is preserved separately: an idle HTTP connection closed during 4k warmup. Remote control requests now use fresh connections, with no mutation retries. The successful rerun uses that fix for every pair.

## Recorded 27B calibration

The [2026-10-12 calibration evidence](../reports/calibration-27b-2026-10-12.json) records real Qwen3.5-27B Q4_K_M prefill and native state export on the RTX 2080. FlashAttention and Q8 cache were enabled. Every request began with an empty prompt cache and evaluated/saved exactly **N−1** tokens. These are native prefill timings; 27B Metal decode and end-to-end split TTFT have not been measured.

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

The earlier 8k/8,192 runs completed their inference trials but exited 139 during GPU telemetry cleanup. Commit `5c5bb4c` fixes the sampler/NVML lifetime race; the later 16k and 32k runs exited 0. An earlier 3,072 candidate failed before inference on TCP port reuse, also fixed. Raw status and failures remain in the evidence rather than being counted as OOMs or hidden. The [calibration guide](calibration.md) gives the guarded Docker recipe, timing definitions and report command. Full interactive graphs and CSVs are saved locally under `.pd-demo/reports/calibration-27b*`.

## Updating the overview

The README chart reads the committed measurement summaries directly. Regenerate it with:

```bash
uv run scripts/render_readme_performance.py
```

The script declares its own plotting dependency; it does not change the demo's runtime dependencies. When adding results, retain the measurement date, settings, paired uncertainty and distinction between measured and modeled outcomes.

