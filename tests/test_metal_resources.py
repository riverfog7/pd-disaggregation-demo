from types import SimpleNamespace

from pd_disaggregation_demo import resources
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import ModelSpec, ResourceSnapshot, RuntimeProfile


def test_mac_metadata_is_cached_and_does_not_invent_dedicated_memory(monkeypatch, tmp_path):
    calls = []
    resources._mac_metadata.cache_clear()
    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        resources.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="Test CPU\n")
    )

    def metal():
        calls.append(True)
        return {"gpu_name": "Test unified GPU", "metal_recommended_bytes": 12 * resources.GiB}

    monkeypatch.setattr(resources, "_query_metal_device", metal)
    try:
        first = resources.capture_resources(tmp_path)
        second = resources.capture_resources(tmp_path)
        assert first.cpu_model == "Test CPU"
        assert first.gpu_name == "Test unified GPU"
        assert first.metal_recommended_bytes == 12 * resources.GiB
        assert second.metal_recommended_bytes == first.metal_recommended_bytes
        assert first.gpu_total_bytes is None
        assert first.gpu_free_bytes is None
        assert first.gpu_used_bytes is None
        assert calls == [True]
    finally:
        resources._mac_metadata.cache_clear()


def test_metal_working_set_admission_separate_from_available_ram(monkeypatch, tmp_path):
    model = ModelSpec(
        model_id="test",
        hf_repo="test/test",
        hf_revision="a" * 40,
        filename="test.gguf",
        sha256="b" * 64,
        size_bytes=resources.GiB,
        layers=8,
        attention_layers=2,
        kv_heads=2,
        head_dim=64,
        recurrent_bytes=4096,
    )
    snapshot = ResourceSnapshot(
        platform="darwin",
        cpu_count=8,
        ram_total_bytes=64 * resources.GiB,
        ram_available_bytes=48 * resources.GiB,
        disk_free_bytes=100 * resources.GiB,
        metal_recommended_bytes=2 * resources.GiB,
    )
    monkeypatch.setattr(resources, "capture_resources", lambda _: snapshot)
    result = resources.check_admission(
        Settings(work_dir=tmp_path), RuntimeProfile(), model, "metal"
    )
    assert not result.allowed
    assert any("Metal working set" in reason for reason in result.reasons)
    assert result.budgets["ram_reserve_bytes"] == 6 * resources.GiB
    assert result.budgets["gpu_reserve_bytes"] == resources.GiB
    assert result.budgets["gpu_required_bytes"] > result.budgets["weights_bytes"]
    snapshot.metal_recommended_bytes = 16 * resources.GiB
    assert resources.check_admission(
        Settings(work_dir=tmp_path), RuntimeProfile(), model, "metal"
    ).allowed


def test_explicit_metal_overcommit_preserves_disk_and_cuda_guards(monkeypatch, tmp_path):
    from pd_disaggregation_demo.catalog import get_model

    snapshot = ResourceSnapshot(
        platform="darwin",
        cpu_count=14,
        ram_total_bytes=48 * resources.GiB,
        ram_available_bytes=resources.GiB,
        disk_free_bytes=100 * resources.GiB,
        metal_recommended_bytes=resources.GiB,
        gpu_free_bytes=resources.GiB,
    )
    monkeypatch.setattr(resources, "capture_resources", lambda _: snapshot)
    monkeypatch.setattr(resources, "_cgroup_available", lambda: None)
    profile = RuntimeProfile(model_id="qwen3.5-27b")
    model = get_model(profile.model_id)
    settings = Settings(work_dir=tmp_path, allow_metal_memory_overcommit=True)
    allowed = resources.check_admission(settings, profile, model, "metal")
    assert allowed.allowed
    assert allowed.budgets["metal_memory_admission_overridden"] == 1
    assert not resources.check_admission(
        Settings(work_dir=tmp_path), profile, model, "metal"
    ).allowed
    cuda = resources.check_admission(settings, profile, model, "cuda")
    assert not cuda.allowed
    assert "metal_memory_admission_overridden" not in cuda.budgets
    assert any("RAM requires" in reason for reason in cuda.reasons)
    assert any("VRAM requires" in reason for reason in cuda.reasons)
    snapshot.disk_free_bytes = resources.GiB
    disk = resources.check_admission(settings, profile, model, "metal")
    assert not disk.allowed
    assert all("Disk requires" in reason for reason in disk.reasons)
