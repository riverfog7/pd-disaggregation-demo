import hashlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from pd_disaggregation_demo import catalog
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import ModelSpec


class Body(httpx.SyncByteStream):
    def __init__(self, data, fail_after=None):
        self.data = data
        self.fail_after = fail_after

    def __iter__(self):
        for offset in range(0, len(self.data), 64 * 1024):
            if self.fail_after is not None and offset >= self.fail_after:
                raise httpx.ReadError("connection interrupted")
            yield self.data[offset : offset + 64 * 1024]


@pytest.fixture
def model(monkeypatch, tmp_path):
    payload = b"bounded GGUF test" * (catalog.DOWNLOAD_CHUNK_BYTES // 8 + 3)
    spec = ModelSpec(
        model_id="test",
        hf_repo="test/public",
        hf_revision="a" * 40,
        filename="test.gguf",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        layers=8,
        attention_layers=2,
        kv_heads=2,
        recurrent_bytes=1024,
    )
    monkeypatch.setitem(catalog.MODELS, "test", spec)
    monkeypatch.setattr(catalog.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(catalog.time, "sleep", lambda _: None)
    settings = Settings(cache_dir=tmp_path, work_dir=tmp_path / "work")
    directory = settings.models_dir / "test"
    directory.mkdir(parents=True)
    return settings, spec, payload, directory / spec.filename


def transport(monkeypatch, handler):
    monkeypatch.setattr(
        catalog,
        "_http_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
    )


def response(payload, *, start=0, status=200, total=None):
    headers = {"Content-Length": str(len(payload))}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{start + len(payload) - 1}/{total}"
    return httpx.Response(status, headers=headers, stream=Body(payload))


def test_download_is_plain_bounded_http_even_if_xet_was_enabled(model, monkeypatch):
    import huggingface_hub
    import huggingface_hub.constants

    settings, spec, payload, final = model
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **kw: pytest.fail("Xet-capable download must not be called"),
    )
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Accept-Encoding"] == "identity"
        assert spec.hf_revision in str(request.url)
        return response(payload)

    transport(monkeypatch, handler)
    assert catalog.ensure_model(settings, "test") == final.resolve()
    assert final.read_bytes() == payload
    assert len(requests) == 1
    assert not final.with_name(final.name + ".part").exists()
    # Cached files are verified without constructing another HTTP client.
    monkeypatch.setattr(catalog, "_http_client", lambda: pytest.fail("cache should not use HTTP"))
    assert catalog.ensure_model(settings, "test") == final.resolve()


def test_existing_partial_resumes_exact_range(model, monkeypatch):
    settings, spec, payload, final = model
    partial = final.with_name(final.name + ".part")
    partial.write_bytes(payload[:53])

    def handler(request):
        assert request.headers["Range"] == "bytes=53-"
        return response(payload[53:], start=53, status=206, total=spec.size_bytes)

    transport(monkeypatch, handler)
    catalog.ensure_model(settings, "test")
    assert final.read_bytes() == payload
    assert not partial.exists()


def test_interrupted_transfer_retries_from_disk_not_from_ram(model, monkeypatch):
    settings, spec, payload, final = model
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200,
                headers={"Content-Length": str(spec.size_bytes)},
                stream=Body(payload, fail_after=catalog.DOWNLOAD_CHUNK_BYTES),
            )
        offset = catalog.DOWNLOAD_CHUNK_BYTES
        assert request.headers["Range"] == f"bytes={offset}-"
        return response(payload[offset:], start=offset, status=206, total=spec.size_bytes)

    transport(monkeypatch, handler)
    catalog.ensure_model(settings, "test")
    assert len(calls) == 2 and final.read_bytes() == payload


def test_server_ignoring_range_restarts_without_duplicate_prefix(model, monkeypatch):
    settings, _, payload, final = model
    final.with_name(final.name + ".part").write_bytes(payload[:53])
    transport(monkeypatch, lambda request: response(payload))
    catalog.ensure_model(settings, "test")
    assert final.read_bytes() == payload


def test_invalid_resume_and_checksum_never_publish_weights(model, monkeypatch):
    settings, spec, payload, final = model
    partial = final.with_name(final.name + ".part")
    partial.write_bytes(payload[:53])
    transport(
        monkeypatch,
        lambda request: response(payload[54:], start=54, status=206, total=spec.size_bytes),
    )
    with pytest.raises(ValueError, match="Content-Range"):
        catalog.ensure_model(settings, "test")
    assert partial.read_bytes() == payload[:53] and not final.exists()
    partial.unlink()
    transport(monkeypatch, lambda request: response(b"x" * len(payload)))
    with pytest.raises(ValueError, match="checksum"):
        catalog.ensure_model(settings, "test")
    assert not partial.exists() and not final.exists()


def test_retryable_http_error_and_disk_reserve_are_bounded(model, monkeypatch):
    settings, _, payload, final = model
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503) if len(attempts) == 1 else response(payload)

    transport(monkeypatch, handler)
    catalog.ensure_model(settings, "test")
    assert len(attempts) == 2
    final.unlink()
    available = iter([100 * 1024**3, 100 * 1024**3, 19 * 1024**3])
    monkeypatch.setattr(
        catalog.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(available))
    )
    transport(monkeypatch, lambda request: response(payload))
    with pytest.raises(ValueError, match="disk reserve"):
        catalog.ensure_model(settings, "test")
    assert not final.exists()
    assert final.with_name(final.name + ".part").stat().st_size == catalog.DOWNLOAD_CHUNK_BYTES


def test_concurrent_fetch_uses_one_http_transfer_and_atomic_result(model, monkeypatch):
    settings, _, payload, final = model
    calls = []

    def handler(request):
        calls.append(request)
        return response(payload)

    transport(monkeypatch, handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: catalog.ensure_model(settings, "test"), range(2)))
    assert results == [final.resolve(), final.resolve()]
    assert len(calls) == 1


def test_complete_partial_finishes_without_network(model, monkeypatch):
    settings, _, payload, final = model
    final.with_name(final.name + ".part").write_bytes(payload)
    transport(monkeypatch, lambda request: pytest.fail("complete partial should not use HTTP"))
    catalog.ensure_model(settings, "test")
    assert final.read_bytes() == payload
