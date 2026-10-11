import asyncio
import json

import httpx

from pd_disaggregation_demo.api import BENCHMARK_JOB_LIMIT, BENCHMARK_REPLAY_LIMIT, create_app
from pd_disaggregation_demo.benchmarks import BenchmarkConfig, BenchmarkEvent, BenchmarkReport
from pd_disaggregation_demo.benchmarks.models import BenchmarkSample
from pd_disaggregation_demo.client.local import LocalClient
from pd_disaggregation_demo.config import Settings
from pd_disaggregation_demo.models import GenerationResult, RunMetrics, RuntimeProfile


class FakePipeline:
    profile = RuntimeProfile()


def replay_events(response):
    return [
        BenchmarkEvent.model_validate_json(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def wait_done(client, job_id):
    for _ in range(100):
        response = await client.get(f"/benchmarks/{job_id}")
        if response.json()["done"]:
            return response.json()
        await asyncio.sleep(0)
    raise AssertionError("Fake benchmark did not finish")


async def test_many_progress_events_retain_one_report_and_bounded_replay(tmp_path, monkeypatch):
    halfway = asyncio.Event()
    resume = asyncio.Event()
    finished = asyncio.Event()
    dumps = []
    original_dump = BenchmarkReport.model_dump

    def counted_dump(self, **kwargs):
        dumps.append(len(self.samples))
        return original_dump(self, **kwargs)

    monkeypatch.setattr(BenchmarkReport, "model_dump", counted_dump)

    async def many_samples(pipeline, config, settings, *, cancel_event):
        report = BenchmarkReport(config=config, environment={"metadata": "x" * 32768})
        for i in range(400):
            mode = "local" if i % 2 == 0 else "split"
            size = 256 if i < 200 else 512
            report.samples.append(
                BenchmarkSample(
                    prompt_tokens=size,
                    repeat=(i % 200) // 2,
                    order=i % 2,
                    result=GenerationResult(
                        mode=mode,
                        model_id=pipeline.profile.model_id,
                        prompt_tokens=size,
                        generated_tokens=128,
                        text="x" * 1024,
                        metrics=RunMetrics(ttft_ms=100, total_ms=200),
                    ),
                )
            )
            for kind in ("progress", "sample"):
                yield BenchmarkEvent(kind=kind, message=str(i), report=report)
            if i == 199:
                halfway.set()
                await resume.wait()
        report.status = "completed"
        yield BenchmarkEvent(kind="complete", report=report)
        finished.set()

    monkeypatch.setattr("pd_disaggregation_demo.benchmarks.run_benchmark", many_samples)
    app = create_app(Settings(work_dir=tmp_path), pipeline=FakePipeline(), with_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        job = (
            await client.post("/benchmarks", json={"repeats": 100, "prompt_sizes": [256, 512]})
        ).json()
        await asyncio.wait_for(halfway.wait(), timeout=2)
        assert not dumps  # Progress does not copy or serialize the growing report.
        current = (await client.get(f"/benchmarks/{job['job_id']}")).json()
        assert not current["done"]
        assert current["status"] == "running"
        assert len(current["report"]["samples"]) == 200
        assert len(current["events"]) == BENCHMARK_REPLAY_LIMIT
        assert current["events_dropped"] == 400 - BENCHMARK_REPLAY_LIMIT
        assert all("report" not in event for event in current["events"])
        assert dumps == [200]

        # Pause a subscriber within an old replay batch while the producer finishes.
        endpoint = next(r.endpoint for r in app.routes if r.path.endswith("/{job_id}/events"))
        response = await endpoint(job["job_id"])
        iterator = response.body_iterator
        first = await anext(iterator)
        resume.set()
        await asyncio.wait_for(finished.wait(), timeout=2)
        chunks = [first] + [chunk async for chunk in iterator]
        live = [
            BenchmarkEvent.model_validate_json(line[6:])
            for chunk in chunks
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]
        assert live[-1].kind == "complete"
        assert len(live[-1].report.samples) == 400
        assert all(event.report is None for event in live[:-1])

        final = await wait_done(client, job["job_id"])
        assert final["status"] == "completed"
        assert len(final["events"]) == BENCHMARK_REPLAY_LIMIT
        assert final["events_dropped"] == 801 - BENCHMARK_REPLAY_LIMIT
        assert len(final["report"]["samples"]) == 400
        assert len(json.dumps(final["events"])) < len(json.dumps(final["report"])) / 2
        late = replay_events(await client.get(job["events_url"]))
        assert len(late) == BENCHMARK_REPLAY_LIMIT
        assert late[-1].kind == "complete"
        assert late[-1].report == BenchmarkReport.model_validate(final["report"])
        assert all(event.report is None for event in late[:-1])
        async with LocalClient("http://test", transport=httpx.ASGITransport(app=app)) as typed:
            received = [
                event
                async for event in typed.benchmark(
                    BenchmarkConfig(prompt_sizes=[256, 512], repeats=100)
                )
            ]
        assert len(received) == BENCHMARK_REPLAY_LIMIT
        assert received[-1].report.status == "completed"
        assert len(received[-1].report.samples) == 400


async def test_old_finished_jobs_pruned_but_saved_reports_remain(tmp_path, monkeypatch):
    started = asyncio.Event()
    resume = asyncio.Event()
    run_ids = []

    async def finished_report(pipeline, config, settings, *, cancel_event):
        report = BenchmarkReport(config=config)
        run_ids.append(report.run_id)
        if len(run_ids) == 1:
            yield BenchmarkEvent(kind="progress", report=report)
            started.set()
            await resume.wait()
        report.status = "completed"
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / f"{report.run_id}.json").write_text(report.model_dump_json())
        yield BenchmarkEvent(kind="complete", report=report)

    monkeypatch.setattr("pd_disaggregation_demo.benchmarks.run_benchmark", finished_report)
    app = create_app(Settings(work_dir=tmp_path), pipeline=FakePipeline(), with_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = (await client.post("/benchmarks", json={})).json()["job_id"]
        await asyncio.wait_for(started.wait(), timeout=2)
        assert (await client.post("/benchmarks", json={})).status_code == 409
        assert (await client.get(f"/benchmarks/{first}")).status_code == 200
        resume.set()
        await wait_done(client, first)
        handles = [first]
        for _ in range(BENCHMARK_JOB_LIMIT + 1):
            handle = (await client.post("/benchmarks", json={})).json()["job_id"]
            await wait_done(client, handle)
            handles.append(handle)
        codes = [(await client.get(f"/benchmarks/{handle}")).status_code for handle in handles]
        assert codes == [404, 404] + [200] * BENCHMARK_JOB_LIMIT
        assert (await client.get(f"/benchmarks/{first}/events")).status_code == 404
        assert len((await client.get("/benchmarks")).json()) == len(handles)
        assert (await client.get(f"/reports/{run_ids[0]}/json")).status_code == 200
        latest = replay_events(await client.get(f"/benchmarks/{handles[-1]}/events"))
        assert latest[-1].report.status == "completed"


async def test_unhandled_job_failure_has_latest_failed_report(tmp_path, monkeypatch):
    async def storage_failure(pipeline, config, settings, *, cancel_event):
        report = BenchmarkReport(config=config, environment={"real_metadata": "retained"})
        yield BenchmarkEvent(kind="progress", report=report)
        raise OSError("Report storage unavailable")

    monkeypatch.setattr("pd_disaggregation_demo.benchmarks.run_benchmark", storage_failure)
    app = create_app(Settings(work_dir=tmp_path), pipeline=FakePipeline(), with_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        handle = (await client.post("/benchmarks", json={})).json()["job_id"]
        status = await wait_done(client, handle)
        assert status["status"] == "failed"
        assert status["report"]["status"] == "failed"
        assert status["report"]["errors"] == ["Report storage unavailable"]
        events = replay_events(await client.get(f"/benchmarks/{handle}/events"))
        assert events[-1].kind == "error"
        assert events[-1].report.environment == {"real_metadata": "retained"}
        assert events[-1].report.status == "failed"
