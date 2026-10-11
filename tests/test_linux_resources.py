from pd_disaggregation_demo import resources
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.models import ModelSpec, ResourceSnapshot, RuntimeProfile


def model():
    return ModelSpec(
        model_id="test",
        hf_repo="test/test",
        hf_revision="f" * 40,
        filename="test.gguf",
        sha256="a" * 64,
        size_bytes=512 * resources.MiB,
        layers=8,
        attention_layers=2,
        kv_heads=2,
        head_dim=64,
        recurrent_bytes=4096,
    )


def snapshot(**kwargs):
    values = dict(
        platform="linux",
        cpu_count=6,
        ram_total_bytes=16 * resources.GiB,
        ram_available_bytes=8 * resources.GiB,
        disk_free_bytes=100 * resources.GiB,
        gpu_free_bytes=5 * resources.GiB,
        gpu_total_bytes=8 * resources.GiB,
    )
    values.update(kwargs)
    return ResourceSnapshot(**values)


def test_admission_reserves_and_never_counts_swap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        resources,
        "capture_resources",
        lambda _: snapshot(
            ram_available_bytes=2 * resources.GiB, swap_used_bytes=8 * resources.GiB
        ),
    )
    monkeypatch.setattr(resources, "_cgroup_available", lambda: None)
    settings = Settings(
        work_dir=tmp_path,
        remote_ram_reserve_bytes=0,
        gpu_reserve_bytes=0,
        disk_reserve_bytes=0,
        skip_admission=True,
    )
    result = resources.check_admission(settings, RuntimeProfile(), model(), "cuda")
    assert not result.allowed
    assert any("swap is not capacity" in reason for reason in result.reasons)
    assert result.budgets["ram_reserve_bytes"] == 2 * resources.GiB
    assert result.budgets["gpu_reserve_bytes"] == resources.GiB
    assert result.budgets["disk_reserve_bytes"] == 20 * resources.GiB


def test_cgroup_budget_does_not_double_reserve_host(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "capture_resources", lambda _: snapshot())
    monkeypatch.setattr(resources, "_cgroup_available", lambda: 2 * resources.GiB)
    result = resources.check_admission(
        Settings(work_dir=tmp_path), RuntimeProfile(), model(), "cuda"
    )
    assert result.allowed
    assert result.budgets["ram_required_bytes"] < 2 * resources.GiB
    monkeypatch.setattr(resources, "_cgroup_available", lambda: resources.MiB)
    assert not resources.check_admission(
        Settings(work_dir=tmp_path), RuntimeProfile(), model(), "cuda"
    ).allowed


def test_cuda_unknown_or_busy_memory_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_cgroup_available", lambda: None)
    for available in (None, resources.GiB):
        monkeypatch.setattr(
            resources, "capture_resources", lambda _: snapshot(gpu_free_bytes=available)
        )
        result = resources.check_admission(
            Settings(work_dir=tmp_path), RuntimeProfile(), model(), "cuda"
        )
        assert not result.allowed
        assert any("GPU" in reason or "VRAM" in reason for reason in result.reasons)


def test_metal_unified_budget_has_six_gib_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "capture_resources", lambda _: snapshot(platform="darwin"))
    settings = Settings(work_dir=tmp_path, local_ram_reserve_bytes=0)
    f16 = resources.check_admission(settings, RuntimeProfile(), model(), "metal")
    q8 = resources.check_admission(settings, RuntimeProfile(cache_type="q8_0"), model(), "metal")
    assert f16.budgets["ram_reserve_bytes"] == 6 * resources.GiB
    assert q8.budgets["kv_bytes"] < f16.budgets["kv_bytes"]
    assert f16.budgets["ram_required_bytes"] > f16.budgets["weights_bytes"]


def test_nonexistent_path_and_cgroup_files(tmp_path):
    result = resources.capture_resources(tmp_path / "does" / "not" / "exist")
    assert result.disk_free_bytes > 0
    assert result.process_rss_bytes > 0
    assert resources._cgroup_available(tmp_path) is None
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("300")
    assert resources._cgroup_available(tmp_path) == 700
    (tmp_path / "memory.max").write_text("max")
    assert resources._cgroup_available(tmp_path) is None
