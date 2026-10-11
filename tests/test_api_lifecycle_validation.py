import importlib.util
from pathlib import Path
from typing import Any

import pytest

from pd_disaggregation_demo.models import GenerationEvent, GenerationResult, RunMetrics

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_api_lifecycle.py"
spec = importlib.util.spec_from_file_location("api_lifecycle_validation", SCRIPT)
assert spec is not None and spec.loader is not None
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


@pytest.fixture
def fake_api(monkeypatch):
    scenario: dict[str, Any] = {
        "outage": False,
        "recover": False,
        "bad_recovery": False,
        "health_calls": 0,
        "cancel_calls": 0,
        "opened": 0,
        "closed": 0,
        "completions": 0,
    }

    class Client:
        def __init__(self, api_url, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

        async def health(self):
            scenario["health_calls"] += 1
            if scenario["recover"] and scenario["health_calls"] > 1:
                scenario["outage"] = False
            return {
                "local": {"status": "ready"},
                "remote": {"status": "unavailable" if scenario["outage"] else "ready"},
            }

        async def cancel(self):
            scenario["cancel_calls"] += 1
            return {"status": "cancelled"}

        async def generate(self, request):
            scenario["opened"] += 1
            try:
                if scenario["outage"] and request.mode == "split":
                    raise RuntimeError(
                        "Remote worker GET /health failed: ReadError; check SSH tunnel"
                    )
                yield GenerationEvent(
                    kind="stage", stage="remote_prefill" if request.mode == "split" else "prefill"
                )
                ids = [10 if request.mode == "local" else 20] * 8
                yield GenerationEvent(kind="token", text="continuation", data={"token_ids": ids})
                scenario["completions"] += 1
                text = "continuation"
                if scenario["bad_recovery"] and scenario["completions"] > 1:
                    text = "different recovery"
                yield GenerationEvent(
                    kind="result",
                    result=GenerationResult(
                        mode=request.mode,
                        model_id="qwen3.5-0.8b",
                        prompt_tokens=10,
                        generated_tokens=8,
                        text=text,
                        metrics=RunMetrics(
                            cache_n=9 if request.mode == "split" else 0,
                            prompt_n=1 if request.mode == "split" else 10,
                        ),
                    ),
                )
            finally:
                scenario["closed"] += 1

    monkeypatch.setattr(validation, "LocalClient", Client)
    return scenario


async def test_api_probe_cancels_and_disconnects_owned_streams_then_checks_recovery(fake_api):
    report = await validation.validate("http://mac", split=True, interruption_delay=0)
    assert report["success"]
    assert {(case["mode"], case["action"]) for case in report["cases"]} == {
        ("local", "cancel"),
        ("local", "disconnect"),
        ("split", "cancel"),
        ("split", "disconnect"),
    }
    assert fake_api["cancel_calls"] == 2
    assert fake_api["closed"] == fake_api["opened"]


async def test_api_probe_detects_different_recovery_continuation(fake_api):
    fake_api["bad_recovery"] = True
    report = await validation.validate("http://mac", interruption_delay=0)
    assert not report["success"]
    assert "differs after interruption" in report["error"]["message"]
    assert fake_api["closed"] == fake_api["opened"]


@pytest.mark.parametrize("recover", [False, True])
async def test_api_probe_observes_external_worker_outage_and_optional_recovery(fake_api, recover):
    fake_api["outage"] = True
    fake_api["recover"] = recover
    report = await validation.validate(
        "http://mac",
        expect_worker_unavailable=True,
        wait_for_worker_recovery=1 if recover else 0,
        split=recover,
        interruption_delay=0,
    )
    assert report["success"]
    assert "ReadError" in report["worker_failure"]["error"]
    assert report["worker_failure"]["local_recovery"]
    assert bool(report["worker_failure"]["split_recovery"]) == recover
    assert fake_api["closed"] == fake_api["opened"]
