"""Cancellation must join background resource reads before releasing their owner."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from pd_disaggregation_demo.calibration import MemoryWatchdog
from pd_disaggregation_demo.config.settings import Settings
from pd_disaggregation_demo.runtime.engine import LlamaRuntime, sample_in_thread


class BlockingSample:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self):
        self.loop.call_soon_threadsafe(self.entered.set)
        try:
            if not self.release.wait(2):
                raise TimeoutError("Test did not release its sampler")
            return {"ram_available_bytes": 64 * 1024**3}
        finally:
            self.finished.set()


async def test_watchdog_joins_slow_sample_before_nvml_shutdown(tmp_path):
    runtime = SimpleNamespace(backend="cuda", settings=Settings(work_dir=tmp_path))
    watchdog = MemoryWatchdog(runtime)
    sample = BlockingSample()
    shutdown = []
    watchdog.sample = sample
    watchdog.gpu = SimpleNamespace(close=lambda: shutdown.append(sample.finished.is_set()))
    watchdog.task = asyncio.create_task(watchdog.run())
    closing = None
    try:
        await asyncio.wait_for(sample.entered.wait(), 1)
        closing = asyncio.create_task(watchdog.close())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.02)
        assert shutdown == []
        sample.release.set()
        await asyncio.wait_for(closing, 1)
        assert shutdown == [True]
    finally:
        sample.release.set()
        await watchdog.close()
        if closing:
            await closing


async def test_native_slot_joins_resource_sample_before_releasing_lock(tmp_path, monkeypatch):
    runtime = LlamaRuntime(Settings(work_dir=tmp_path))
    sample = BlockingSample()
    calls = 0

    def capture(_path):
        nonlocal calls
        calls += 1
        data = sample() if calls > 1 else {"ram_available_bytes": 64 * 1024**3}
        return SimpleNamespace(model_dump=lambda **_kwargs: data)

    monkeypatch.setattr("pd_disaggregation_demo.resources.capture_resources", capture)

    async def erase():
        pass

    monkeypatch.setattr(runtime, "_erase", erase)

    async def operation():
        async with runtime._slot():
            await sample.entered.wait()

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(sample.entered.wait(), 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.02)
        assert runtime._lock.locked()
        assert runtime._active is task
        sample.release.set()
        await asyncio.wait_for(task, 1)
        assert sample.finished.is_set()
        assert not runtime._lock.locked()
        assert runtime._active is None
    finally:
        sample.release.set()
        await asyncio.wait_for(task, 1)


@pytest.mark.parametrize("sample_fails", [False, True])
async def test_repeated_cancellation_still_joins_thread_and_preserves_cancellation(sample_fails):
    sample = BlockingSample()

    def measure():
        result = sample()
        if sample_fails:
            raise RuntimeError("Sampling also failed")
        return result

    task = asyncio.create_task(sample_in_thread(measure))
    try:
        await asyncio.wait_for(sample.entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.02)
        sample.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert sample.finished.is_set()
    finally:
        sample.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
