import asyncio
import json
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pd_disaggregation_demo.api import create_app
from pd_disaggregation_demo.cli import app
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import (
    GenerationEvent,
    GenerationResult,
    RunMetrics,
    RuntimeProfile,
)


class FakePipeline:
    profile = RuntimeProfile(model_id="qwen3.5-0.8b")
    started = False
    cancelled = False

    @asynccontextmanager
    async def session(self):
        yield

    async def start(self):
        self.started = True

    async def close(self):
        self.started = False

    async def health(self):
        return {"status": "ready"}

    async def format_chat(self, messages):
        return messages[-1]["content"]

    async def cancel(self):
        self.cancelled = True

    async def generate(self, request):
        yield GenerationEvent(kind="stage", stage="prefill")
        yield GenerationEvent(kind="token", text="hello", token_id=12)
        yield GenerationEvent(
            kind="result",
            result=GenerationResult(
                mode=request.mode,
                model_id=self.profile.model_id,
                prompt_tokens=4,
                generated_tokens=1,
                text="hello",
                metrics=RunMetrics(ttft_ms=10),
            ),
        )


def test_api_stream_and_chat_share_pipeline(tmp_path):
    pipeline = FakePipeline()
    settings = Settings(work_dir=tmp_path)
    app = create_app(settings, pipeline=pipeline, with_ui=False)
    with TestClient(app) as client:
        assert pipeline.started
        assert client.get("/health").json()["status"] == "ready"
        stream = client.post("/generate", json={"prompt": "test"})
        assert "event: token" in stream.text
        assert '"ttft_ms": 10.0' in stream.text
        chat = {
            "model": pipeline.profile.model_id,
            "messages": [{"role": "user", "content": "test"}],
        }
        response = client.post("/v1/chat/completions", json=chat)
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        stream = client.post("/v1/chat/completions", json={**chat, "stream": True})
        assert "[DONE]" in stream.text
        chunks = [
            json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: {")
        ]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert client.post("/cancel").status_code == 200
        assert pipeline.cancelled
    assert not pipeline.started


def test_api_rejects_wrong_models_and_missing_artifacts(tmp_path):
    with TestClient(
        create_app(Settings(work_dir=tmp_path), pipeline=FakePipeline(), with_ui=False)
    ) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "wrong", "messages": [{"role": "user", "content": "hello"}]},
            ).status_code
            == 400
        )
        assert client.get("/artifacts/bad/manifest").status_code == 400
        assert client.get(f"/artifacts/{'a' * 32}/state").status_code == 404


def test_nonstream_cancellation_is_actionable_without_internal_server_error(tmp_path):
    class CancelledPipeline(FakePipeline):
        async def configure(self, profile):
            raise asyncio.CancelledError()

        async def prefill(self, request):
            raise asyncio.CancelledError()

    pipeline = CancelledPipeline()
    with TestClient(
        create_app(Settings(work_dir=tmp_path), pipeline=pipeline, with_ui=False)
    ) as client:
        for endpoint, payload in (
            ("/configure", pipeline.profile.model_dump()),
            ("/prefill", {"prompt": "Test cancellation"}),
        ):
            response = client.post(endpoint, json=payload)
            assert response.status_code == 409
            assert "cancelled" in response.json()["detail"]
        assert client.get("/health").json()["status"] == "ready"


def test_cli_catalog_and_launch_help():
    runner = CliRunner()
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 4
    for command in [
        ["serve", "mac", "--help"],
        ["serve", "linux", "--help"],
        ["benchmark", "--help"],
        ["deploy", "--help"],
    ]:
        assert runner.invoke(app, command).exit_code == 0
