"""Only the demo's Compose project is managed; source travels exclusively via Git."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from ..config.settings import Settings
from ..models import RuntimeProfile
from ..resources import GiB, capture_resources

PROJECT = "pd-disaggregation-demo"
SHA = re.compile(r"^[0-9a-f]{40}$")


def _run(
    args: list[str], *, cwd: Path | None = None, input: str | None = None, timeout: float = 120
) -> str:
    result = subprocess.run(
        args, cwd=cwd, input=input, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} failed ({result.returncode}): {(result.stderr or result.stdout)[-5000:]}"
        )
    return result.stdout


def _ssh(settings: Settings, script: str, *, timeout: float = 120) -> str:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            settings.remote_ssh_host,
            "bash -s",
        ],
        input=script,
        timeout=timeout,
    )


def _repo(settings: Settings) -> Path:
    return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=settings.work_dir.parent).strip())


def _clean_revision(repo: Path, revision: str | None = None) -> str:
    if _run(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("Deploy requires a clean committed checkout, including untracked files")
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    revision = revision or head
    if not SHA.fullmatch(revision):
        raise ValueError("Deploy requires an exact full 40-character commit SHA")
    if (
        _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=repo).strip()
        != revision
    ):
        raise ValueError("Revision does not identify an exact commit")
    return revision


def _remote_paths(settings: Settings) -> tuple[str, str]:
    path = settings.remote_repo
    if not path.startswith("/") or "\n" in path or path == "/":
        raise ValueError("remote_repo must be a non-root absolute path")
    return shlex.quote(path), shlex.quote(path + "/.deploy")


def inspect_remote(settings: Settings) -> dict:
    """Read-only host resources and demo Docker status; never touch other workloads."""
    _remote_paths(settings)
    script = f"""set -eu
python3 - <<'PYREMOTE'
import json, os, pathlib, shutil, subprocess
mem = dict((line.split(':')[0], int(line.split()[1])*1024)
           for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
def command(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=15)
    return {{'returncode':p.returncode, 'stdout':p.stdout.strip(), 'stderr':p.stderr.strip()}}
print(json.dumps({{'ram_total_bytes':mem['MemTotal'],
 'ram_available_bytes':mem['MemAvailable'],
 'swap_used_bytes':mem['SwapTotal']-mem['SwapFree'],
 'disk_free_bytes':shutil.disk_usage(os.path.expanduser('~')).free,
 'gpu':command(['nvidia-smi','--id=0','--query-gpu=name,memory.total,memory.free,memory.used',
                '--format=csv,noheader,nounits']),
 'docker':command(['docker','version','--format','{{{{json .Server}}}}']),
 'worker':command(['docker','ps','-a','--filter','label=com.docker.compose.project={PROJECT}',
                  '--filter','label=com.docker.compose.service=worker',
                  '--format','{{{{json .}}}}'])}}))
PYREMOTE
"""
    return json.loads(_ssh(settings, script))


def monitor(settings: Settings) -> dict:
    """Report resources plus stats/top only for containers owned by this demo."""
    snapshot = inspect_remote(settings)
    output = _ssh(
        settings,
        f"""set -eu
ids=$(docker ps --filter label=com.docker.compose.project={PROJECT} \\
  --filter label=com.docker.compose.service=worker --format '{{{{.ID}}}}')
if [ -n "$ids" ]; then
  docker stats --no-stream --format '{{{{json .}}}}' $ids
  for id in $ids; do docker top "$id" -eo pid,ppid,rss,comm; done
fi
""",
    )
    snapshot["own_worker_stats_and_processes"] = output
    return snapshot


def open_tunnel(
    settings: Settings, *, local_port: int | None = None, ssh_host: str | None = None
) -> subprocess.Popen:
    """Caller owns returned foreground SSH process and terminates it on shutdown."""
    port = local_port or settings.worker_port
    if not 1 <= port <= 65535:
        raise ValueError("Invalid local tunnel port")
    return subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{settings.worker_port}",
            ssh_host or settings.remote_ssh_host,
        ],
        stdin=subprocess.DEVNULL,
    )


def _health_script(port: int, revision_expression: str, timeout: float) -> str:
    # Expression is an internally controlled shell variable, never raw user text.
    return f"""python3 - {port} "{revision_expression}" {float(timeout)} <<'PYHEALTH'
import json, sys, time, urllib.request
port, revision, timeout = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
deadline = time.monotonic() + timeout
last = 'No health response'
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{{port}}/health', timeout=5) as r:
            health = json.load(r)
        if health.get('git_revision') == revision and health.get('status') in ('ok','ready'):
            print(json.dumps(health))
            sys.exit(0)
        last = str(health)
    except Exception as error:
        last = str(error)
    time.sleep(2)
raise SystemExit('Worker failed health/revision check: '+last)
PYHEALTH
"""


def deploy(
    settings: Settings,
    revision: str,
    *,
    health_timeout: float = 900,
    profile: RuntimeProfile | None = None,
) -> dict:
    """Push/fetch exact clean SHA, build release, switch and verify; restore on failure."""
    local = _repo(settings)
    revision = _clean_revision(local, revision)
    origin = _run(["git", "remote", "get-url", "origin"], cwd=local).strip()
    if "\n" in origin or origin.startswith("-"):
        raise ValueError("Invalid Git origin URL")
    _run(
        ["git", "push", "origin", f"{revision}:refs/heads/pd-deploy/{revision}"],
        cwd=local,
        timeout=180,
    )
    repo, root = _remote_paths(settings)
    worker_profile = profile or RuntimeProfile()
    profile_json = worker_profile.model_dump_json()
    script = f"""set -eu
repo={repo}
root={root}
revision={shlex.quote(revision)}
# uid 1000 is the unprivileged runtime user; only demo-owned asset paths are changed.
test "$(id -u)" = 1000 || {{ echo 'Demo data owner must have uid 1000' >&2; exit 1; }}
if [ ! -e "$repo/.git" ]; then
  git clone --no-checkout {shlex.quote(origin)} "$repo"
fi
mkdir -p "$root/releases" "$root/env" "$root/data/cache" "$root/data/work"
git -C "$repo" fetch origin "refs/heads/pd-deploy/$revision"
test "$(git -C "$repo" rev-parse FETCH_HEAD)" = "$revision"
release="$root/releases/$revision"
if [ ! -e "$release" ]; then git -C "$repo" worktree add --detach "$release" "$revision"; fi
test "$(git -C "$release" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$release" status --porcelain)"
python3 - "$root/env/$revision.env" "$revision" "$root/data" \\
  {settings.worker_port} {shlex.quote(profile_json)} <<'PYENV'
import json, pathlib, shutil, sys
mem = dict((line.split(':')[0], int(line.split()[1])*1024)
           for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
reserve = 2*1024**3
available = mem['MemAvailable']
# CUDA compilation uses one job and requires at least 1GiB above the host reserve.
if available < reserve + 1024**3:
    raise SystemExit('Insufficient RAM for safe single-job CUDA build')
if shutil.disk_usage(pathlib.Path(sys.argv[1]).parent).free < 20*1024**3 + 8*1024**3:
    raise SystemExit('Insufficient disk for image, model/state and 20GiB reserve')
limit = min(6*1024**3, available-reserve)
profile=json.loads(sys.argv[5])
values={{'PDD_RELEASE_SHA':sys.argv[2], 'PDD_REMOTE_DATA':sys.argv[3],
 'PDD_WORKER_MEMORY':str(limit), 'PDD_WORKER_CPUS':'2', 'PDD_BUILD_JOBS':'1',
 'PDD_WORKER_PORT':sys.argv[4], 'PDD_MODEL_ID':profile['model_id'],
 'PDD_CONTEXT_SIZE':str(profile['context_size']), 'PDD_CACHE_TYPE':profile['cache_type'],
 'PDD_WORKER_PROFILE':json.dumps(profile,separators=(',',':'))}}
pathlib.Path(sys.argv[1]).write_text(''.join(k+'='+json.dumps(v)+'\\n' for k,v in values.items()))
PYENV
compose() {{
  docker compose --project-name {PROJECT} --env-file "$root/env/$revision.env" \\
    -f "$release/compose.yaml" "$@"
}}
# Build leaves the previous healthy deployment serving throughout compilation.
compose build worker
prior=$(readlink "$root/current" || true)
restore() {{
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    if [ -n "$prior" ]; then
      old=$(basename "$prior")
      docker compose --project-name {PROJECT} --env-file "$root/env/$old.env" \\
        -f "$prior/compose.yaml" up -d --no-build worker >&2 || true
      {_health_script(settings.worker_port, "$old", min(health_timeout, 300))}
    else
      compose stop worker >&2 || true
    fi
  fi
  exit "$rc"
}}
trap restore EXIT
compose up -d --no-build worker
container=$(compose ps -q worker)
label='{{{{index .Config.Labels "org.opencontainers.image.revision"}}}}'
test "$(docker inspect --format "$label" "$container")" = "$revision"
{_health_script(settings.worker_port, "$revision", health_timeout)}
if [ -n "$prior" ] && [ "$prior" != "$release" ]; then ln -sfn "$prior" "$root/previous"; fi
ln -sfn "$release" "$root/current"
trap - EXIT
"""
    output = _ssh(settings, script, timeout=max(3600, health_timeout + 300))
    # Git/build progress can coexist with the final JSON health line.
    health = next(
        json.loads(line) for line in reversed(output.splitlines()) if line.startswith("{")
    )
    return {
        "revision": revision,
        "health": health,
        "release": settings.remote_repo + "/.deploy/releases/" + revision,
    }


def rollback(settings: Settings, *, health_timeout: float = 900) -> dict:
    """Switch only this Compose worker to the saved previous committed release."""
    _, root = _remote_paths(settings)
    output = _ssh(
        settings,
        f"""set -eu
root={root}
previous=$(readlink "$root/previous")
current=$(readlink "$root/current")
revision=$(basename "$previous")
test -z "$(git -C "$previous" status --porcelain)"
test "$(git -C "$previous" rev-parse HEAD)" = "$revision"
docker compose --project-name {PROJECT} --env-file "$root/env/$revision.env" \\
  -f "$previous/compose.yaml" up -d --no-build worker
{_health_script(settings.worker_port, "$revision", health_timeout)}
ln -sfn "$previous" "$root/current"
ln -sfn "$current" "$root/previous"
""",
        timeout=health_timeout + 120,
    )
    return {
        "health": next(
            json.loads(line) for line in reversed(output.splitlines()) if line.startswith("{")
        )
    }


def _local_compose(settings: Settings, profile: RuntimeProfile | None = None) -> list[str]:
    repo = _repo(settings)
    env_path = settings.work_dir / "worker-compose.env"
    if profile is not None:
        revision = _clean_revision(repo)
        resources = capture_resources(settings.work_dir)
        available = resources.ram_available_bytes - max(2 * GiB, settings.remote_ram_reserve_bytes)
        if available < GiB:
            raise ValueError("Insufficient host RAM above the 2GiB reserve")
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        data = settings.work_dir / "worker-data"
        (data / "cache").mkdir(parents=True, exist_ok=True)
        (data / "work").mkdir(parents=True, exist_ok=True)
        if hasattr(os, "getuid") and os.getuid() != 1000:
            raise ValueError("Linux Compose asset paths require the runtime uid 1000 owner")
        values = {
            "PDD_RELEASE_SHA": revision,
            "PDD_REMOTE_DATA": str(data.resolve()),
            "PDD_WORKER_MEMORY": str(min(available, 6 * GiB)),
            "PDD_WORKER_PORT": str(settings.worker_port),
            "PDD_BUILD_JOBS": str(min(max(1, settings.build_jobs), 2)),
            "PDD_WORKER_PROFILE": profile.model_dump_json(),
            "PDD_MODEL_ID": profile.model_id,
            "PDD_CONTEXT_SIZE": str(profile.context_size),
            "PDD_CACHE_TYPE": profile.cache_type,
        }
        env_path.write_text("".join(k + "=" + json.dumps(v) + "\n" for k, v in values.items()))
    if not env_path.is_file():
        raise ValueError("No managed Linux worker deployment exists; run serve-linux first")
    return [
        "docker",
        "compose",
        "--project-name",
        PROJECT,
        "--env-file",
        str(env_path),
        "-f",
        str(repo / "compose.yaml"),
    ]


def serve_linux(settings: Settings, profile: RuntimeProfile, detach: bool = True) -> dict:
    args = _local_compose(settings, profile)
    up = ["up", "--build"] + (["-d"] if detach else []) + ["worker"]
    output = _run(args + up, timeout=3600)
    return {"status": "started", "output": output, "port": settings.worker_port}


def stop_linux(settings: Settings) -> dict:
    return {"status": "stopped", "output": _run(_local_compose(settings) + ["stop", "worker"])}


def status_linux(settings: Settings) -> dict:
    return {
        "output": _run(_local_compose(settings) + ["ps", "--all", "--format", "json", "worker"])
    }
