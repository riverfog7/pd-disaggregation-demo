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

    def ssh(settings, script, **kwargs):
        scripts.append(script)
        check = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert check.returncode == 0, check.stderr
        return '{"status":"ready","git_revision":"' + sha + '"}\n'

    monkeypatch.setattr(operations, "_ssh", ssh)
    settings = Settings(work_dir=tmp_path, remote_repo="/home/demo/space path/demo")
    result = operations.deploy(settings, sha)
    assert result["revision"] == sha
    assert any(args[:3] == ["git", "push", "origin"] for args in calls)
    script = scripts[0]
    assert "worktree add --detach" in script
    assert "trap restore EXIT" in script
    assert script.index("compose build worker") < script.index("compose up -d")
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
