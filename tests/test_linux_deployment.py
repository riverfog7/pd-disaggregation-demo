import ast
import re
import subprocess

import pytest

from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.deployment import operations


def test_reject_dirty_and_abbreviated_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(operations, "_run", lambda *a, **k: " M modified.py\n")
    with pytest.raises(ValueError, match="clean"):
        operations._clean_revision(tmp_path, "a" * 40)
    monkeypatch.setattr(operations, "_run", lambda args, **k: "" if "status" in args else "a" * 40)
    with pytest.raises(ValueError, match="40-character"):
        operations._clean_revision(tmp_path, "abc")


def test_deploy_git_only_transaction_and_shell_syntax(monkeypatch, tmp_path):
    sha = "a" * 40
    monkeypatch.setattr(operations, "_repo", lambda _: tmp_path)
    monkeypatch.setattr(operations, "_clean_revision", lambda *a: sha)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return "https://github.com/test/demo.git" if "get-url" in args else ""

    monkeypatch.setattr(operations, "_run", run)
    scripts = []
    options = []

    def ssh(settings, script, **kwargs):
        scripts.append(script)
        options.append(kwargs)
        check = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert check.returncode == 0, check.stderr
        for marker, code in re.findall(r"<<'(PY[A-Z]+)'\n(.*?)\n\1", script, re.S):
            ast.parse(code, filename=marker)
        return '{"status":"ready","git_revision":"' + sha + '"}\n'

    monkeypatch.setattr(operations, "_ssh", ssh)
    settings = Settings(work_dir=tmp_path, remote_repo="/home/demo/space path/demo")
    result = operations.deploy(settings, sha)
    assert result["revision"] == sha
    assert options[0]["timeout"] == (
        operations.DEFAULT_BUILD_TIMEOUT_SECONDS + settings.request_timeout + 900 + 300
    )
    assert options[0]["stream"]
    assert any(args[:3] == ["git", "push", "origin"] for args in calls)
    script = scripts[0]
    assert "worktree add --detach" in script
    assert "trap restore EXIT" in script
    assert script.index('compose build --builder "$builder" worker') < script.index("compose up -d")
    assert '"memory=$limit,memory-swap=$limit"' in script
    assert "max-parallelism = 1" in script
    assert "def _build_budget(available: int, reserve: int)" in script
    assert "build_limit, build_jobs = _build_budget(available," in script
    assert "\'PDD_BUILD_JOBS\':str(build_jobs)" in script
    assert 'docker buildx stop "$builder"' in script
    assert "--use" not in script
    assert "rsync" not in script and "scp" not in script
    assert "org.opencontainers.image.revision" in script
    assert "SwapFree" not in script


def test_tunnel_is_owned_foreground_and_loopback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        operations.subprocess, "Popen", lambda args, **kw: calls.append(args) or 123
    )
    assert operations.open_tunnel(Settings(), local_port=18090, ssh_host="lan") == 123
    assert "127.0.0.1:18090:127.0.0.1:8090" in calls[0]
    assert "-f" not in calls[0]
    with pytest.raises(ValueError):
        operations.open_tunnel(Settings(), local_port=65536)


def test_monitor_filters_own_project_and_service(monkeypatch):
    monkeypatch.setattr(operations, "inspect_remote", lambda _: {})
    scripts = []
    monkeypatch.setattr(operations, "_ssh", lambda _, script: scripts.append(script) or "stats")
    assert operations.monitor(Settings())["own_worker_stats_and_processes"] == "stats"
    assert "label=com.docker.compose.project=pd-disaggregation-demo" in scripts[0]
    assert "label=com.docker.compose.service=worker" in scripts[0]
    assert 'docker top "$id"' in scripts[0]


@pytest.mark.parametrize("available,limit,jobs", [(5, 2, 1), (7, 4, 2)])
def test_local_builder_is_capped_and_stopped_on_failure(
    monkeypatch, tmp_path, available, limit, jobs
):
    from pd_disaggregation_demo.models import ResourceSnapshot

    monkeypatch.setattr(
        operations,
        "capture_resources",
        lambda _: ResourceSnapshot(
            platform="linux",
            cpu_count=6,
            ram_total_bytes=16 * operations.GiB,
            ram_available_bytes=available * operations.GiB,
            disk_free_bytes=100 * operations.GiB,
        ),
    )
    monkeypatch.setattr(operations, "_repo", lambda _: tmp_path)
    monkeypatch.setattr(operations, "_clean_revision", lambda _: "a" * 40)
    monkeypatch.setattr(
        operations.subprocess, "run", lambda *a, **kw: type("Result", (), {"returncode": 1})()
    )
    (tmp_path / "builder-name").write_text("pd-demo-build\n")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["compose-test", "build"]:
            assert kwargs["timeout"] == operations.DEFAULT_BUILD_TIMEOUT_SECONDS
            assert kwargs["stream"]
            assert f"BUILD_JOBS={jobs}" in args
            raise RuntimeError("compile OOM")
        return ""

    monkeypatch.setattr(operations, "_run", run)
    with pytest.raises(RuntimeError, match="OOM"):
        operations._bounded_local_build(Settings(work_dir=tmp_path), ["compose-test"])
    assert f"memory={limit * operations.GiB},memory-swap={limit * operations.GiB}" in calls[0]
    assert "--use" not in calls[0]
    assert calls[-1][:3] == ["docker", "buildx", "stop"]
    assert (tmp_path / "buildkitd.toml").read_text() == "[worker.oci]\n  max-parallelism = 1\n"


def test_failed_manual_rollback_has_restore_trap(monkeypatch):
    scripts = []

    def ssh(settings, script, **kwargs):
        scripts.append(script)
        checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert checked.returncode == 0, checked.stderr
        return '{"status":"ready"}\n'

    monkeypatch.setattr(operations, "_ssh", ssh)
    operations.rollback(Settings())
    assert "trap restore_current EXIT" in scripts[0]
    assert 'current_revision=$(basename "$current")' in scripts[0]
    assert scripts[0].rfind("trap - EXIT") > scripts[0].rfind("ln -sfn")


def test_streaming_diagnostics_keeps_json_stdout_clean(capfd):
    import sys

    script = "import sys; print('build phase 1',file=sys.stderr,flush=True); print('{\"ok\":true}')"
    output = operations._run([sys.executable, "-c", script], stream=True)
    captured = capfd.readouterr()
    assert output.strip() == '{"ok":true}'
    assert captured.out == ""
    assert captured.err.strip() == "build phase 1"


def test_builder_cache_identity_persists_and_adopts_only_owned_cache(monkeypatch, tmp_path):
    name = "pd-demo-" + "b" * 40
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return "default\nother-user\n" + name + "\n"

    monkeypatch.setattr(operations, "_run", run)
    assert operations._owned_builder(tmp_path) == name
    assert operations._owned_builder(tmp_path) == name
    assert len(calls) == 1
    (tmp_path / "builder-name").write_text("other-user")
    with pytest.raises(ValueError, match="owned"):
        operations._owned_builder(tmp_path)


def test_fresh_builder_uses_stable_project_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(operations, "_run", lambda *a, **kw: "default\nother-user\n")
    assert operations._owned_builder(tmp_path) == "pd-demo-build"


def test_occupied_worker_port_refuses_unrelated_listener_before_build(monkeypatch):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        monkeypatch.setattr(operations, "_run", lambda *a, **kw: "")
        with pytest.raises(ValueError, match="unrelated listener"):
            operations._check_worker_port(Settings(worker_port=port))

        # A known own worker publishing this exact port can be replaced safely.
        def owned(args, **kwargs):
            if args[1] == "ps":
                return "own-container"
            return (
                '[{"NetworkSettings":{"Ports":{"8090/tcp":[{"HostIp":"127.0.0.1","HostPort":"'
                + str(port)
                + '"}]}}}]'
            )

        monkeypatch.setattr(operations, "_run", owned)
        operations._check_worker_port(Settings(worker_port=port))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_build_timeout_fails_before_any_external_operation(monkeypatch, timeout):
    monkeypatch.setattr(operations, "_repo", lambda _: pytest.fail("must validate before Git"))
    with pytest.raises(ValueError, match="Build timeout"):
        operations.deploy(Settings(), "a" * 40, build_timeout=timeout)


def test_interrupted_first_remote_deploy_adopts_cache_without_current_symlink(
    monkeypatch, tmp_path
):
    import os
    import sys

    sha = "d" * 40
    builder = "pd-demo-" + sha
    monkeypatch.setattr(operations, "_repo", lambda _: tmp_path)
    monkeypatch.setattr(operations, "_clean_revision", lambda *a: sha)
    monkeypatch.setattr(operations, "_run", lambda *a, **kw: "https://github.com/test/demo.git")
    scripts = []
    monkeypatch.setattr(
        operations, "_ssh", lambda _, script, **kw: scripts.append(script) or '{"status":"ready"}\n'
    )
    operations.deploy(Settings(work_dir=tmp_path), sha)
    code = re.search(r"<<'PYBUILDER'\n(.*?)\nPYBUILDER", scripts[0], re.S).group(1)
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'test "$1" = buildx && test "$2" = ls || exit 9\n'
        "printf '%s\\n' default other-user '" + builder + " *'\n"
    )
    docker.chmod(0o755)
    record = tmp_path / "builder-name"
    current = tmp_path / "current"
    assert not current.exists() and not current.is_symlink()
    result = subprocess.run(
        [sys.executable, "-", str(record), str(current)],
        input=code,
        env={**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"]},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == builder
    assert record.read_text().strip() == builder


@pytest.mark.parametrize(
    "available,reserve,limit,jobs",
    [
        (3 * operations.GiB, 2 * operations.GiB, operations.GiB, 1),
        (6 * operations.GiB, 2 * operations.GiB, 2 * operations.GiB, 1),
        (7 * operations.GiB - 1, 2 * operations.GiB, 2 * operations.GiB, 1),
        (7 * operations.GiB, 2 * operations.GiB, 4 * operations.GiB, 2),
        (12 * operations.GiB, 2 * operations.GiB, 4 * operations.GiB, 2),
        (8 * operations.GiB, 4 * operations.GiB, 2 * operations.GiB, 1),
        (9 * operations.GiB, 4 * operations.GiB, 4 * operations.GiB, 2),
    ],
)
def test_build_budget_threshold_and_remote_policy_match(available, reserve, limit, jobs):
    import inspect

    # The SSH script receives this exact dependency-free function rather than
    # maintaining a second independently changing policy implementation.
    namespace = {"GiB": operations.GiB}
    exec(inspect.getsource(operations._build_budget), namespace)
    assert operations._build_budget(available, reserve) == (limit, jobs)
    assert namespace["_build_budget"](available, reserve) == (limit, jobs)
    assert limit <= available - max(reserve, 2 * operations.GiB)


def test_build_budget_never_borrows_host_reserve():
    with pytest.raises(ValueError, match="Insufficient RAM"):
        operations._build_budget(3 * operations.GiB - 1, 0)
