# CUDA worker deployment interfaces

All functions are exported from `pd_disaggregation_demo.deployment`. They are
synchronous except the worker app lifecycle; call blocking operations with
`asyncio.to_thread` in an async API.

- `inspect_remote(settings) -> dict`: read-only RAM/swap/disk/GPU/Docker inventory.
- `monitor(settings) -> dict`: inventory plus stats and process RSS for only the
  `pd-disaggregation-demo` Compose project's `worker` containers.
- `deploy(settings, revision, *, health_timeout=900, profile=None) -> dict`: exact
  full 40-character commit SHA, clean local source required. Pushes an immutable
  SHA-named Git release branch, fetches it, creates a detached remote release
  worktree, builds before switching, checks image SHA label and HTTP health SHA,
  and restores the prior deployment if the switch fails. Models and state remain
  in independent persistent asset directories, never copied with source.
- `rollback(settings, *, health_timeout=900) -> dict`: starts the saved previous
  release and verifies health before updating current/previous pointers.
- `open_tunnel(settings, *, local_port=None, ssh_host=None) -> Popen`: foreground
  SSH process owned by the caller, loopback bind on both ends. Terminate and wait
  on it at caller shutdown; use `settings.lan_ssh_host` explicitly when desired.
- `serve_linux(settings, profile, detach=True) -> dict`, `stop_linux(settings) ->
  dict`, `status_linux(settings) -> dict`: manage only the demo Compose worker on
  a Linux host. Source must be clean before a build. Default detached launch
  returns after Compose starts the worker; readiness is exposed by `/health`.

`backend.app.create_worker_app(settings, profile) -> FastAPI` starts and closes
`LlamaRuntime(..., backend="cuda")` in its lifespan. The image entrypoint is
`pd-demo worker`. That CLI command must read `PDD_WORKER_PROFILE` JSON when set,
so every runtime placement/context/cache field survives deployment.

Remote release paths are `$remote_repo/.deploy/releases/<sha>`, with environment
files in `.deploy/env` and persistent asset binds in `.deploy/data/{cache,work}`.
The runtime user and host asset owner use UID 1000. No unrelated containers or
processes are stopped. Failed first deployments stop only their own worker.

The image pins llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`, CUDA 12.8.1,
and SM75. The build has one CUDA compilation job by default (maximum two). Only
llama-server, native libraries and Python 3.13 package dependencies enter the
runtime image; Gradio/Plotly extras and developer tools are omitted.

Compose uses CDI `nvidia.com/gpu=all` and publishes only host loopback port 8090.
Runtime RAM is capped from measured host availability minus a 2GiB reserve,
with memory+swap equal to memory (no container swap). Deployment refuses build
headroom below 1GiB above that reserve and disk below 28GiB. The engine admission
check separately enforces host, cgroup, GPU and model/state/compute budgets.
A dedicated persistent `docker-container` BuildKit builder also has a hard RAM
cap of `min(host_available - 2GiB, 2GiB)`, equal memory+swap limit, two CPU quota,
and `[worker.oci] max-parallelism=1`. Compose uses that explicit builder; the
user default builder is not changed. Builder registration is refreshed with
`--keep-state` to update limits while retaining its own cache. The builder name
is saved in `.deploy/builder-name` (local serve uses its work directory), reused
across code revisions, and defaults to `pd-demo-build` on a fresh host. Existing
demo SHA builders are adopted under their current name to preserve their state
volume; unrelated builders are excluded. The builder stops
after each build, including failures, before starting the worker. A compilation
OOM fails only the constrained builder and leaves the prior worker serving.
[Docker driver limits](https://docs.docker.com/build/builders/drivers/docker-container/)
and [BuildKit parallelism](https://docs.docker.com/build/buildkit/configure/) describe
the underlying controls. Do not run builds while host headroom is changing sharply.

Admission conservatively counts the full mmap weight startup peak, live cache,
exported state and compute; swap never counts as capacity. Minimum reserves are
1GiB CUDA VRAM, 2GiB Linux host RAM, 6GiB Metal unified RAM, and 20GiB disk.
A resource rejection is HTTP 422 with `detail.code="admission_refused"`;
operational failures remain separate. `skip_admission` does not override these
minimum reserves. Live worker state downloads read at most 1MiB per chunk, and
only canonical UUID artifact directories are accepted. Export checksums and sizes
are checked before publication; expired state is reaped and in-use transfers are
protected from deletion.

The separate `models` Compose service runs once with `pd-demo models fetch
--model <selected-id>` before a worker switch. It has no GPU devices or published
port and only the asset cache writable. Serving workers mount `/data/cache/models`
read-only. To switch to another model, prefetch it with this asset service (or
redeploy with the selected profile) before `/configure`; the worker verifies the
existing pinned file and never downloads into its read-only model directory.

CPU FFN admission does not estimate tensor placement from parameter count: it
keeps the full conservative GPU weight budget and also budgets all weights on
host. Large profiles such as 27B can therefore be rejected even with CPU FFN
placement. A safe future relaxation requires measured architecture-specific
weight residency, compute/cache peaks and mmap behavior; this release does not
claim that an unmeasured placement is safe.

The worker watchdog samples host RAM, GPU free memory and its own cgroup every
0.5 seconds. Three consecutive samples below the host/GPU reserve (or below
64MiB cgroup allocation headroom) cancel only the active demo request and close
its owned native engine. Health becomes `resource_refused` with
`error.code="resource_pressure"` and the triggering snapshot. New prefill requests
are rejected until explicit `/configure` restarts the runtime through admission.
A transient single sample does not stop a request. The worker never restarts
itself repeatedly or stops an unrelated workload to recover capacity.

Deployment and local Linux launch probe the requested host loopback port before
building. An occupied port is accepted only when Docker identifies the demo's
own worker publishing that exact address/port; unrelated listeners are left
running and the build is refused. Long build/deployment commands stream their
diagnostics to stderr immediately while retaining stdout for structured JSON.
