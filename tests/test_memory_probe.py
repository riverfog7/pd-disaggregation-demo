import json
import subprocess
from pathlib import Path

import pytest

from pd_disaggregation_demo.config.settings import ENGINE_REVISION, Settings
from pd_disaggregation_demo.models import RuntimeProfile
from pd_disaggregation_demo.runtime import build, memory_probe
from pd_disaggregation_demo.runtime.arguments import model_context_arguments
from pd_disaggregation_demo.runtime.engine import LlamaRuntime
from pd_disaggregation_demo.runtime.memory_probe import MiB, estimate_memory, parse_memory_estimate


def test_memory_output_conservatively_bounds_native_truncated_mib():
    raw = "CUDA0 4000 256 512\nMTL0 3 4 5\nHost 12000 0 13\n"
    estimate = parse_memory_estimate(raw, "diagnostics")
    assert estimate.devices["CUDA0"].model_bytes == 4001 * MiB
    assert estimate.devices["CUDA0"].total_bytes == (4000 + 256 + 512 + 3) * MiB
    assert estimate.host.context_bytes == MiB
    assert estimate.host.reported_mib == (12000, 0, 13)
    assert estimate.stdout == raw and estimate.stderr == "diagnostics"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "CUDA0 1 2 3\n",
        "Host 1 2\n",
        "Host -1 2 3\n",
        "Host nan 2 3\n",
        "Host 1 2 3\nHost 4 5 6\n",
        "Host 1 2 3\nCUDA0 nope 1 2\n",
        "warning: incomplete\nHost 1 2 3\n",
    ],
)
def test_memory_output_rejects_malformed_or_ambiguous_rows(raw):
    with pytest.raises(ValueError):
        parse_memory_estimate(raw)


def test_estimator_uses_same_model_context_flags_as_worker(monkeypatch, tmp_path):
    profile = RuntimeProfile(
        cache_type="q8_0",
        context_size=16384,
        batch_size=8192,
        ubatch_size=2048,
        cpu_ffn=64,
        gpu_layers=-1,
        threads=6,
    )
    settings = Settings(work_dir=tmp_path)
    model = tmp_path / "model.gguf"
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "CUDA0 1 2 3\nHost 4 5 6\n", "log")

    monkeypatch.setattr(memory_probe.subprocess, "run", run)
    estimate = estimate_memory(Path("llama-fit-params"), profile, "cuda", model, timeout=17)
    common = model_context_arguments(profile, "cuda", model)
    worker = LlamaRuntime(settings, profile, "cuda")._arguments(Path("llama-server"), model)
    assert worker[1 : 1 + len(common)] == common
    assert list(estimate.arguments) == ["llama-fit-params", *common, "--fit-print", "on"]
    assert calls[0][1] == {"capture_output": True, "text": True, "timeout": 17, "check": True}
    assert "--fit" in common and common[common.index("--fit") + 1] == "off"
    assert "--op-offload" in common
    assert "--slots" not in estimate.arguments and "--cache-ram" not in estimate.arguments
    assert "--ctx-checkpoints" not in estimate.arguments and "--spec-type" not in estimate.arguments


def test_requested_device_must_be_present(monkeypatch):
    monkeypatch.setattr(
        memory_probe.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 0, "Host 1 2 3\n", ""),
    )
    with pytest.raises(ValueError, match="CUDA0"):
        estimate_memory(Path("helper"), RuntimeProfile(), "cuda", Path("model"))


def test_cpu_placement_never_inherits_gpu_layer_request():
    arguments = model_context_arguments(RuntimeProfile(gpu_layers=20), "cpu", Path("model"))
    assert arguments[arguments.index("--gpu-layers") + 1] == "0"
    assert arguments[arguments.index("--device") + 1] == "none"


def test_cached_server_only_builds_missing_estimator_without_reconfiguring(monkeypatch, tmp_path):
    settings = Settings(cache_dir=tmp_path)
    directory = tmp_path / "engines" / ENGINE_REVISION / "build" / "metal"
    binary = directory / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"original server")
    (directory / "pd-build.json").write_text(
        json.dumps({"revision": ENGINE_REVISION, "backend": "metal"})
    )
    calls = []
    monkeypatch.setattr(build, "_run", lambda arguments, cwd=None: calls.append(arguments) or "")
    assert build.ensure_engine(settings, "metal") == binary
    assert len(calls) == 1
    assert calls[0] == [
        "cmake",
        "--build",
        str(directory),
        "--target",
        "llama-fit-params",
        "-j",
        str(settings.build_jobs),
    ]
    assert binary.read_bytes() == b"original server"


def test_configured_server_missing_helper_has_explicit_error(monkeypatch, tmp_path):
    settings = Settings(engine_binary=tmp_path / "llama-server")
    monkeypatch.setattr(build, "_run", lambda arguments, cwd=None: ENGINE_REVISION[:7])
    with pytest.raises(FileNotFoundError, match="beside server"):
        build.ensure_memory_probe(settings, "cuda")


def test_estimator_must_report_pinned_revision(monkeypatch, tmp_path):
    server = tmp_path / "llama-server"
    server.with_name(build.SERVER_PROBE_NAME).touch()
    monkeypatch.setattr(build, "ensure_engine", lambda settings, backend: server)
    monkeypatch.setattr(build, "_run", lambda arguments, cwd=None: "wrong revision")
    with pytest.raises(ValueError, match="pinned revision"):
        build.ensure_memory_probe(Settings(engine_binary=server), "metal")


def test_probe_forwards_calibration_tensor_overrides(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(memory_probe, "ensure_memory_probe", lambda *a: Path("helper"))

    def fake_estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return parse_memory_estimate("Host 100 1 1\nCUDA0 100 1 1\n")

    monkeypatch.setattr(memory_probe, "estimate_memory", fake_estimate)
    overrides = (r"^blk\.[0-9]+\.(attn_qkv|attn_gate|ssm_out)\.weight$=CPU",)
    settings, profile = Settings(), RuntimeProfile()
    memory_probe.probe_memory(
        settings, profile, "cuda", tmp_path / "model", timeout=3, tensor_overrides=overrides
    )
    assert calls == [
        ((Path("helper"), profile, "cuda", tmp_path / "model", 3), {"tensor_overrides": overrides})
    ]


def server_provenance(profile, **overrides):
    return {
        "kind": build.SERVER_PROBE_KIND,
        "engine_revision": ENGINE_REVISION,
        "linked_commit": ENGINE_REVISION[:9],
        "probe_source_sha256": build.server_probe_source_sha256(),
        "parallel": 1,
        "n_outputs_max": 1,
        "n_outputs_max_per_seq": 1,
        "n_rs_seq": 0,
        "n_ctx": profile.context_size,
        "n_batch": profile.batch_size,
        "n_ubatch": profile.ubatch_size,
        **overrides,
    }


def test_server_probe_uses_explicit_non_speculative_command_and_provenance(monkeypatch):
    profile = RuntimeProfile(context_size=8320, batch_size=8192, ubatch_size=8192)
    calls = []
    metadata = server_provenance(profile)

    def run(arguments, **kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments,
            0,
            "CUDA0 1900 430 2500\nHost 14000 0 14\n",
            "PD_SERVER_FIT_PARAMS " + json.dumps(metadata) + "\n",
        )

    monkeypatch.setattr(memory_probe.subprocess, "run", run)
    estimate = estimate_memory(
        Path(build.SERVER_PROBE_NAME),
        profile,
        "cuda",
        Path("model"),
        tensor_overrides=(r"^output\.weight$=CPU",),
    )
    args = calls[0]
    assert args[args.index("--spec-type") + 1] == "none"
    assert args[args.index("--fit") + 1] == "off"
    assert args[args.index("--fit-print") + 1] == "on"
    assert args[args.index("--override-tensor") + 1] == r"^output\.weight$=CPU"
    assert estimate.provenance["n_outputs_max"] == 1
    assert estimate.provenance["kind"] == build.SERVER_PROBE_KIND
    assert estimate.provenance["probe_source_sha256"] == build.server_probe_source_sha256()


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_outputs_max": 8192},
        {"n_outputs_max_per_seq": 2},
        {"parallel": 4},
        {"n_rs_seq": 8},
        {"engine_revision": "wrong"},
        {"linked_commit": "wrong"},
        {"probe_source_sha256": "stale"},
        {"n_ubatch": 512},
        {"kind": "generic-fit-params"},
    ],
)
def test_server_probe_rejects_wrong_context_or_provenance(monkeypatch, overrides):
    profile = RuntimeProfile()
    metadata = server_provenance(profile, **overrides)
    monkeypatch.setattr(
        memory_probe.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments,
            0,
            "CUDA0 1 2 3\nHost 4 5 6\n",
            "PD_SERVER_FIT_PARAMS " + json.dumps(metadata) + "\n",
        ),
    )
    with pytest.raises(ValueError, match="provenance|libraries"):
        estimate_memory(Path(build.SERVER_PROBE_NAME), profile, "cuda", Path("model"))


def test_server_probe_does_not_accept_rows_without_provenance(monkeypatch):
    monkeypatch.setattr(
        memory_probe.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, "CUDA0 1 2 3\nHost 4 5 6\n", ""
        ),
    )
    with pytest.raises(ValueError, match="provenance"):
        estimate_memory(Path(build.SERVER_PROBE_NAME), RuntimeProfile(), "cuda", Path("model"))


def test_server_probe_cached_build_adds_only_its_target(monkeypatch, tmp_path):
    settings = Settings(cache_dir=tmp_path)
    directory = tmp_path / "engines" / ENGINE_REVISION / "build" / "metal"
    server = directory / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"unchanged-server")
    helper = server.with_name(build.SERVER_PROBE_NAME)
    calls = []
    monkeypatch.setattr(build, "ensure_engine", lambda *_: server)

    def run(arguments, cwd=None):
        calls.append(arguments)
        if arguments[:2] == ["cmake", "--build"]:
            helper.touch()
        if "--version" in arguments:
            return (
                f"{build.SERVER_PROBE_KIND} engine={ENGINE_REVISION} "
                f"linked_commit={ENGINE_REVISION[:9]} "
                f"source_sha256={build.server_probe_source_sha256()}"
            )
        return ""

    monkeypatch.setattr(build, "_run", run)
    assert build.ensure_memory_probe(settings, "metal") == helper
    assert len(calls) == 3
    assert calls[0][-1] == f"-DCMAKE_PROJECT_INCLUDE={build.SERVER_PROBE_SOURCE / 'inject.cmake'}"
    assert calls[1] == [
        "cmake",
        "--build",
        str(directory),
        "--target",
        build.SERVER_PROBE_NAME,
        "-j",
        str(settings.build_jobs),
    ]
    assert server.read_bytes() == b"unchanged-server"
    calls.clear()
    assert build.ensure_memory_probe(settings, "metal") == helper
    assert calls == [[str(helper), "--version"]]
