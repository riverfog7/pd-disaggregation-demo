# Shared implementation contract

All package code is in src/pd_disaggregation_demo. Parent owns config/, models.py,
catalog.py, cli.py, api.py, pyproject.toml, uv.lock, README and integration tests.
Do not change another owner's files without coordinating. Make atomic commits.

## Runtime and handoff owner

Own runtime/, client/, artifacts.py, pipeline.py and focused tests.
Use the models and Settings already present. Async interfaces:

- runtime.engine.LlamaRuntime(settings, profile=RuntimeProfile(), backend=None)
- await runtime.start(), close(), configure(profile), health(), tokenize(prompt)
- await runtime.prefill(token_ids: list[int]) -> PrefillArtifact
- runtime.generate(request: GenerationRequest) -> AsyncIterator[GenerationEvent]
- runtime.decode(artifact: PrefillArtifact, request: DecodeRequest) -> same iterator
- runtime.profile and runtime.fingerprint attributes
- runtime.build.ensure_engine(settings, backend) -> Path (sync)
- artifacts.load_artifact(directory: Path) -> PrefillArtifact and validate_artifact
- Pipeline(settings, profile=RuntimeProfile()); same lifecycle/profile, tokenize,
  generate, decode, health; prefill(PrefillRequest) -> PrefillArtifact.
- Pipeline.configure(profile) applies local config; remote is configured on demand
  for split to the same model/cache/context, retaining remote placement settings.
- Pipeline.cancel() cancels only the active demo request and releases its slot.
- Pipeline owns the one-request lock. Engine also protects its single native slot.

The parent supplies catalog.get_model(id) -> ModelSpec and
catalog.ensure_model(settings, id) -> Path (sync; run in asyncio.to_thread).
The Linux owner supplies resources.capture_resources(path: Path) -> ResourceSnapshot
and resources.check_admission(settings, profile, model, backend) -> AdmissionResult,
whose .allowed bool, .reasons list[str], .model_dump() expose decisions. Call before load.

Remote API: GET /health, /capabilities; POST /configure (RuntimeProfile JSON);
POST /prefill (PrefillRequest JSON with full canonical token_ids) -> ArtifactManifest;
GET /artifacts/{id}/manifest -> ArtifactManifest; GET /artifacts/{id}/state -> binary;
DELETE /artifacts/{id}; POST /cancel. Full prompt N -> prefix N-1 -> save -> transfer
-> restore -> evaluate last token. Require n_saved/n_restored/cache_n N-1, prompt_n 1.
Use native llama-server APIs and native state files. Artifact directories contain
manifest.json and state.bin; transfer with bounded chunks and checksums.
Do not implement kernels/cache tensor conversion. No silent local fallback.

## Linux/deployment owner

Own backend/, deployment/, resources.py, Dockerfile, compose.yaml, .dockerignore,
and focused tests. backend.app.create_worker_app(settings, profile) -> FastAPI,
lifespan creates LlamaRuntime backend=cuda. Backend uses above remote endpoints.
Health includes status, profile, fingerprint, git_revision, resources.
Expose deployment API with straightforward functions for parent CLI; document them.
Docker worker entrypoint is `pd-demo worker` (parent implements command).
Image must contain exact engine SHA, CUDA 12.8 SM75 build, package via uv, no UI deps.
CDI device nvidia.com/gpu=all; service loopback host port8090, internal port8090.
Use persistent models and state, runtime launch API, cgroup memory+swap limits,
1GiB VRAM/2GiB host RAM reserves, safe Git fetch/exact-SHA release worktrees/rollback.
Do not stop unrelated processes. Do not deploy until parent asks after integration.

## Benchmark and UI owner

Own benchmarks/, ui/ and focused tests. Pipeline contract above is source of truth.
ui.app.build_ui(pipeline, settings) -> Gradio Blocks for parent to mount at /demo.
benchmarks exports BenchmarkConfig and async run_benchmark(pipeline, config, settings)
iterator of structured progress events; provide a collect helper for CLI/API.
Implement run storage, JSON/CSV/self-contained HTML reports, paired stats/bootstrap CI,
alternating local/split, fresh prompt cache and warm weights; startup separate.
Default matrix 1k/4k/8k/16k/32k,128 output,5 repeats; admit contexts correctly including
generated tokens and reload both runtimes when needed. Report skipped unsafe sizes.
Benchmark UI is primary: TTFT/prefill/decode, phase charts, model/cache/context controls,
history comparisons, export, cancel. Chat secondary. Do not create separate frontend.
Use Gradio + Plotly and async events. Do not fake measurements or silently fall back.

All owners: add meaningful focused tests, report checks and limitations to parent.
Current source is main initial commit + shared contracts. Memory pressure is real;
avoid large model tests on the remote before parent coordinates resource admission.
