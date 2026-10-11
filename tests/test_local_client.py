import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from pd_disaggregation_demo import cli
from pd_disaggregation_demo.benchmarks import BenchmarkConfig, BenchmarkEvent, BenchmarkReport
from pd_disaggregation_demo.client.local import LocalClient
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import (
    ArtifactManifest,
    DecodeRequest,
    GenerationEvent,
    GenerationRequest,
    GenerationResult,
    RunMetrics,
    RuntimeFingerprint,
    RuntimeProfile,
)


def manifest():
    return ArtifactManifest(
        artifact_id="a" * 32,
        model_id="qwen3.5-0.8b",
        token_ids=[1, 2],
        prefix_tokens=1,
        fingerprint=RuntimeFingerprint(
            engine_revision="pinned",
            model_sha256="b" * 64,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            context_size=4096,
        ),
        state_bytes=123,
        state_sha256="c" * 64,
        source_backend="metal",
    )


def generation_response():
    events = [
        GenerationEvent(kind="stage", stage="prefill"),
        GenerationEvent(kind="token", text="test", token_id=10),
        GenerationEvent(
            kind="result",
            result=GenerationResult(
                mode="local",
                model_id="qwen3.5-0.8b",
                prompt_tokens=2,
                generated_tokens=1,
                text="test",
                metrics=RunMetrics(ttft_ms=5),
            ),
        ),
    ]
    return sse([event.model_dump(mode="json") for event in events])


def sse(events):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        text=": keepalive\n\n"
        + "".join(f"event: progress\ndata: {json.dumps(event)}\n\n" for event in events),
    )


def fake_api(calls, artifact=None, report_status="completed"):
    report = BenchmarkReport(
        config=BenchmarkConfig(prompt_sizes=[256], repeats=1), status=report_status
    )

    def handler(request):
        data = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, data))
        path = request.url.path
        if path == "/configure":
            return httpx.Response(200, json={"status": "ready"})
        if path in ("/generate", "/decode"):
            return generation_response()
        if path == "/prefill" or path.endswith("/manifest"):
            return httpx.Response(200, json=(artifact or manifest()).model_dump(mode="json"))
        if path == "/benchmarks":
            return httpx.Response(202, json={"job_id": "d" * 32})
        if path.endswith("/events"):
            kind = {"completed": "complete", "failed": "error", "cancelled": "cancelled"}[
                report_status
            ]
            return sse([BenchmarkEvent(kind=kind, report=report).model_dump(mode="json")])
        if path.startswith("/benchmarks/"):
            return httpx.Response(
                200, json={"done": True, "report": report.model_dump(mode="json")}
            )
        if path == "/cancel":
            return httpx.Response(200, json={"status": "cancelled"})
        raise AssertionError(f"Unexpected endpoint: {path}")

    return httpx.MockTransport(handler)


def patch_client(monkeypatch, transport):
    import pd_disaggregation_demo.client.local as local

    monkeypatch.setattr(
        local,
        "LocalClient",
        lambda url, **kwargs: LocalClient(
            url,
            **kwargs,
            transport=transport,
        ),
    )

    async def forbidden(*_args):
        raise AssertionError("API mode started a second native engine")

    monkeypatch.setattr(cli, "_operation", forbidden)


async def test_typed_client_generation_decode_and_prefill():
    calls = []
    async with LocalClient("http://127.0.0.1:8080", transport=fake_api(calls)) as client:
        await client.configure(RuntimeProfile(model_id="qwen3.5-0.8b"))
        generated = [event async for event in client.generate(GenerationRequest(prompt="hello"))]
        decoded = [event async for event in client.decode("a" * 32, DecodeRequest(max_tokens=1))]
        assert generated[-1].result.text == "test"
        assert decoded[-1].result.metrics.ttft_ms == 5
        assert (await client.manifest("a" * 32)).artifact_id == "a" * 32
    assert calls[0][1] == "/configure"
    assert calls[2][2]["artifact_id"] == "a" * 32


def test_cli_api_generate_never_starts_native_and_preserves_profile(monkeypatch):
    calls = []
    patch_client(monkeypatch, fake_api(calls))
    result = CliRunner().invoke(
        cli.app,
        [
            "generate",
            "--api-url",
            "http://127.0.0.1:8080",
            "--prompt",
            "hello",
            "--model",
            "qwen3.5-0.8b",
            "--context",
            "4096",
            "--cache",
            "q8_0",
            "--max-tokens",
            "1",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert [path for _, path, _ in calls] == ["/configure", "/generate"]
    assert calls[0][2]["model_id"] == "qwen3.5-0.8b"
    assert calls[0][2]["cache_type"] == "q8_0"
    assert calls[0][2]["context_size"] == 4096
    assert json.loads(result.stdout.splitlines()[-1])["kind"] == "result"


def test_cli_api_prefill_returns_server_owned_manifest(monkeypatch):
    calls = []
    artifact = manifest()
    patch_client(monkeypatch, fake_api(calls, artifact))
    result = CliRunner().invoke(
        cli.app, ["prefill", "--api-url", "http://127.0.0.1:8080", "--prompt", "hello", "--remote"]
    )
    assert result.exit_code == 0, result.exception
    output = json.loads(result.stdout)
    assert output["artifact_id"] == artifact.artifact_id
    assert output["manifest"] == artifact.model_dump(mode="json")
    assert "directory" not in output
    assert calls[-1][2]["backend"] == "remote"


def test_cli_api_decode_needs_only_local_manifest_and_server_id(monkeypatch, tmp_path):
    calls = []
    artifact = manifest()
    (tmp_path / "manifest.json").write_text(artifact.model_dump_json())
    patch_client(monkeypatch, fake_api(calls, artifact))
    result = CliRunner().invoke(
        cli.app,
        ["decode", str(tmp_path), "--api-url", "http://127.0.0.1:8080", "--max-tokens", "1"],
    )
    assert result.exit_code == 0, result.exception
    assert not (tmp_path / "state.bin").exists()
    assert [path for _, path, _ in calls] == [
        "/configure",
        f"/artifacts/{'a' * 32}/manifest",
        "/decode",
    ]
    assert calls[0][2]["model_id"] == artifact.model_id
    assert calls[0][2]["context_size"] == 4096
    assert calls[0][2]["cache_type"] == "q8_0"
    assert calls[-1][2]["artifact_id"] == artifact.artifact_id


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 1), ("cancelled", 1)])
def test_cli_api_benchmark_uses_job_configuration_and_terminal_status(
    monkeypatch, status, exit_code
):
    calls = []
    patch_client(monkeypatch, fake_api(calls, report_status=status))
    result = CliRunner().invoke(
        cli.app,
        ["benchmark", "--api-url", "http://127.0.0.1:8080", "--sizes", "256", "--repeats", "1"],
    )
    assert result.exit_code == exit_code, result.exception
    assert calls[0][1] == "/benchmarks"
    assert calls[0][2]["prompt_sizes"] == [256]
    assert not any(path == "/configure" for _, path, _ in calls)
    assert json.loads(result.stdout.splitlines()[-1])["report"]["status"] == status


async def test_ctrl_c_cancellation_posts_cancel_before_client_close(monkeypatch, tmp_path):
    calls = []
    patch_client(monkeypatch, fake_api(calls))

    async def interrupted(_client):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cli._api_operation(
            "http://127.0.0.1:8080", Settings(work_dir=tmp_path), None, interrupted
        )
    assert [path for _, path, _ in calls] == ["/cancel"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, headers={"content-type": "text/event-stream"}, text="data: {}"),
        sse([{"kind": "stage", "stage": "prefill"}]),
        sse([{"error": "worker unavailable"}]),
        httpx.Response(409, json={"detail": "one active request"}),
    ],
)
async def test_api_errors_or_truncated_streams_fail_without_fallback(response):
    async with LocalClient(
        "http://127.0.0.1:8080", transport=httpx.MockTransport(lambda _: response)
    ) as client:
        with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
            _ = [event async for event in client.generate(GenerationRequest(prompt="hello"))]


@pytest.mark.parametrize(
    "url", ["", "ftp://localhost", "http://user:secret@localhost", "http://localhost/?q=1"]
)
def test_api_url_validation(url):
    with pytest.raises(ValueError):
        LocalClient(url)


def test_default_cli_path_still_uses_native_operation(monkeypatch):
    called = []

    async def native(settings, profile, _operation):
        called.append(profile.model_id)

    monkeypatch.setattr(cli, "_operation", native)
    result = CliRunner().invoke(cli.app, ["generate", "--prompt", "hello"])
    assert result.exit_code == 0
    assert called == ["qwen3.5-4b"]


def test_cli_api_http_failure_and_empty_url_do_not_start_native(monkeypatch):
    patch_client(monkeypatch, httpx.MockTransport(lambda _: httpx.Response(409)))
    runner = CliRunner()
    for url in ("http://127.0.0.1:8080", ""):
        result = runner.invoke(cli.app, ["generate", "--api-url", url, "--prompt", "test"])
        assert result.exit_code != 0
        assert not isinstance(result.exception, AssertionError)


def test_api_mode_help_is_available_for_all_four_commands():
    for command in ("generate", "prefill", "decode", "benchmark"):
        result = CliRunner().invoke(cli.app, [command, "--help"])
        assert result.exit_code == 0
        assert "api-url" in result.stdout


async def test_local_client_uses_actual_fastapi_generation_endpoints(tmp_path):
    from pd_disaggregation_demo.api import create_app

    class ExistingServer:
        started = False
        profile = RuntimeProfile(model_id="qwen3.5-0.8b")

        async def configure(self, profile):
            self.profile = profile

        async def health(self):
            return {"status": "ready"}

        async def generate(self, request):
            yield GenerationEvent(kind="token", text="hello", token_id=12)
            yield GenerationEvent(
                kind="result",
                result=GenerationResult(
                    mode=request.mode,
                    model_id=self.profile.model_id,
                    prompt_tokens=2,
                    generated_tokens=1,
                    text="hello",
                    metrics=RunMetrics(ttft_ms=10),
                ),
            )

    server = ExistingServer()
    app = create_app(Settings(work_dir=tmp_path), pipeline=server, with_ui=False)
    async with LocalClient("http://mac-api", transport=httpx.ASGITransport(app=app)) as client:
        await client.configure(RuntimeProfile(model_id="qwen3.5-0.8b"))
        events = [event async for event in client.generate(GenerationRequest(prompt="hello"))]
    assert events[-1].result.text == "hello"
    assert events[-1].result.metrics.ttft_ms == 10
    assert not server.started
