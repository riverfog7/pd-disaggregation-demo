import asyncio
import contextlib
import json

import gradio as gr

from ..benchmarks import BenchmarkConfig, list_reports, run_benchmark
from ..benchmarks.charts import curves, memory_chart, memory_rows, phases, transfer
from ..models import GenerationRequest, RuntimeProfile

MODELS = [
    ("Qwen3.5 0.8B · quick check", "qwen3.5-0.8b"),
    ("Qwen3.5 4B · default", "qwen3.5-4b"),
    ("Qwen3.5 9B", "qwen3.5-9b"),
    ("Qwen3.5 27B", "qwen3.5-27b"),
]
SUMMARY_HEADERS = [
    "Prompt tokens",
    "Local TTFT ms",
    "Split TTFT ms",
    "Speedup",
    "95% CI low",
    "95% CI high",
    "Complete pairs",
    "Supported win",
]


def _summary_rows(report):
    return [
        [
            row.get(key)
            for key in (
                "prompt_tokens",
                "local_ttft_ms",
                "split_ttft_ms",
                "speedup",
                "ci_low",
                "ci_high",
                "pairs",
                "supported_win",
            )
        ]
        for row in report.summary
    ]


def _cards(report):
    if not report.samples:
        return "<p>Measurements appear after the first completed request.</p>"
    sample = report.samples[-1]
    metrics = sample.result.metrics
    values = [
        ("TTFT", metrics.ttft_ms, "ms"),
        ("Prefill", metrics.prefill_tps, "tokens/s"),
        ("Decode", metrics.decode_tps, "tokens/s"),
        ("Save", metrics.export_ms, "ms"),
        ("Transfer", metrics.transfer_ms, "ms"),
        ("Restore", metrics.restore_ms, "ms"),
    ]
    items = "".join(
        '<div style="flex:1;min-width:130px;padding:14px;border:1px solid #ddd;border-radius:10px">'
        f'<div>{label}</div><strong style="font-size:24px">'
        f"{value:,.2f}</strong> {unit}</div>"
        if value is not None
        else f'<div style="padding:14px">{label}: unavailable</div>'
        for label, value, unit in values
    )
    return (
        f"<p>Latest request: {sample.prompt_tokens:,} tokens · pair {sample.repeat + 1} · "
        f"{sample.result.mode}</p><div style='display:flex;flex-wrap:wrap;gap:10px'>{items}</div>"
    )


def _text_history(history):
    messages = []
    for message in history or []:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        messages.append({"role": message["role"], "content": str(content)})
    return messages


def build_ui(pipeline, settings) -> gr.Blocks:
    """Build the main benchmark screen; parent mounts it on its FastAPI lifespan."""
    active = {"cancel": None}

    def history_choices():
        return [
            (
                f"{r.created_at:%m-%d %H:%M} · {r.config.label or r.config.profile.model_id} · "
                f"{r.status} · {r.run_id[:8]}",
                r.run_id,
            )
            for r in list_reports(settings.reports_dir)
        ]

    def profile_from(model, cache, context, batch, ubatch):
        # Preserve the runtime placement settings chosen by the CLI.
        base = pipeline.profile.model_dump()
        base.update(
            model_id=model,
            cache_type=cache,
            context_size=int(context),
            batch_size=int(batch),
            ubatch_size=int(ubatch),
        )
        return RuntimeProfile(**base)

    async def benchmark(model, cache, context, batch, ubatch, sizes, output, repeats, label):
        try:
            prompt_sizes = [int(s.strip()) for s in sizes.replace(" ", ",").split(",") if s.strip()]
            config = BenchmarkConfig(
                profile=profile_from(model, cache, context, batch, ubatch),
                prompt_sizes=prompt_sizes,
                output_tokens=int(output),
                repeats=int(repeats),
                label=label,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        cancel_event = asyncio.Event()
        active["cancel"] = cancel_event
        closing = False
        try:
            async with contextlib.aclosing(
                run_benchmark(pipeline, config, settings, cancel_event=cancel_event)
            ) as stream:
                async for event in stream:
                    report = event.report
                    if report is None:
                        continue
                    done = len(report.samples)
                    total = len(config.prompt_sizes) * config.repeats * 2
                    skipped = len(report.skipped) * config.repeats * 2
                    percent = 100 * (done + skipped) / total
                    status = (
                        f"**{event.stage or event.kind}** · {event.message}\n\n"
                        f"Measured {done}/{total} requests · skipped {len(report.skipped)} sizes "
                        f"· {percent:.0f}%"
                    )
                    if report.errors:
                        status += "\n\n" + "\n".join(f"Failure: {e}" for e in report.errors)
                    if report.skipped:
                        status += "\n\n" + "\n".join(
                            f"Skipped {s.prompt_tokens:,}: {s.reason}" for s in report.skipped
                        )
                    try:
                        yield (
                            status,
                            _cards(report),
                            curves([report]),
                            phases(report),
                            transfer(report),
                            _summary_rows(report),
                            memory_rows(report),
                            memory_chart(report),
                            report.model_dump(mode="json"),
                            list(report.exports.values()) or None,
                            gr.update(choices=history_choices()),
                            gr.update(interactive=report.status != "running"),
                        )
                    except (GeneratorExit, asyncio.CancelledError):
                        closing = True
                        raise
        except Exception as exc:
            if closing:
                raise
            yield (
                f"**Failed** · {exc}",
                *(gr.skip() for _ in range(10)),
                gr.update(interactive=True),
            )
        finally:
            active["cancel"] = None

    async def cancel():
        if active["cancel"] is not None:
            active["cancel"].set()
        await pipeline.cancel()
        return "Cancellation requested; the partial report will be saved."

    def refresh():
        return gr.update(choices=history_choices())

    def compare(run_ids):
        selected = set(run_ids or [])
        reports = [r for r in list_reports(settings.reports_dir) if r.run_id in selected]
        return curves(reports), [path for r in reports for path in r.exports.values()]

    async def chat(prompt, history, mode, model, cache, context, batch, ubatch, output):
        async with pipeline.session():
            async with contextlib.aclosing(
                chat_request(prompt, history, mode, model, cache, context, batch, ubatch, output)
            ) as stream:
                async for update in stream:
                    yield update

    async def chat_request(prompt, history, mode, model, cache, context, batch, ubatch, output):
        try:
            profile = profile_from(model, cache, context, batch, ubatch)
            await pipeline.configure(profile)
            messages = _text_history(history) + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
            yield messages, "Preparing request"
            # Conversation is explicitly re-evaluated so it shares the empty-cache runtime policy.
            conversation = await pipeline.format_chat(messages[:-1])
            async with contextlib.aclosing(
                pipeline.generate(
                    GenerationRequest(
                        prompt=conversation,
                        mode=mode,
                        max_tokens=int(output),
                        temperature=0,
                    )
                )
            ) as stream:
                async for event in stream:
                    if event.kind == "token":
                        messages[-1]["content"] += event.text
                        yield messages, "Decoding"
                    elif event.kind == "stage":
                        yield messages, event.stage
                    elif event.kind == "result" and event.result:
                        messages[-1]["content"] = event.result.text
                        yield messages, json.dumps(event.result.metrics.model_dump(), indent=2)
                    elif event.kind == "error":
                        yield messages, f"Failed: {event.text or event.data}"
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Prefill / Decode · Measured benchmark", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Prefill / Decode\nCompare **Mac local prefill → Mac decode** with "
            "**CUDA remote prefill → Mac decode**, using paired, measured requests."
        )
        with gr.Row():
            model = gr.Dropdown(MODELS, value=pipeline.profile.model_id, label="Model")
            cache = gr.Radio(["f16", "q8_0"], value=pipeline.profile.cache_type, label="KV cache")
        with gr.Accordion("Context and batch configuration", open=False):
            with gr.Row():
                context = gr.Number(
                    value=pipeline.profile.context_size,
                    precision=0,
                    minimum=256,
                    maximum=262144,
                    label="Minimum context tokens",
                )
                batch = gr.Number(
                    value=pipeline.profile.batch_size,
                    precision=0,
                    minimum=1,
                    label="Mac batch tokens",
                )
                ubatch = gr.Number(
                    value=pipeline.profile.ubatch_size,
                    precision=0,
                    minimum=1,
                    label="Mac microbatch tokens",
                )
            gr.Markdown(
                "Each size is admitted with context ≥ prompt + output. Unsafe sizes are "
                "recorded as skipped. Loading or reloading weights is reported separately. "
                "Model, cache and context match on both machines; CUDA keeps the batch settings "
                "chosen when launching its worker. Reports record both profiles."
            )
        with gr.Tab("Benchmark"):
            with gr.Row():
                sizes = gr.Textbox(
                    value="1024,4096,8192,16384,32768", label="Prompt sizes (tokens)"
                )
                output = gr.Number(
                    value=128, precision=0, minimum=1, maximum=8192, label="Output tokens"
                )
                repeats = gr.Number(
                    value=5, precision=0, minimum=1, maximum=100, label="Paired repeats"
                )
                label = gr.Textbox(label="Run label", placeholder="Optional comparison label")
            with gr.Row():
                run = gr.Button("Run paired benchmark", variant="primary")
                stop = gr.Button("Cancel active request", variant="stop")
            status = gr.Markdown("Ready. No measurements yet.")
            cards = gr.HTML("<p>TTFT, prefill, decode and handoff measurements appear here.</p>")
            chart = gr.Plot(label="Latency and throughput")
            gr.Markdown(
                "Medians summarize each mode. Speedup uses complete within-repeat pairs with "
                "a 95% bootstrap interval. A supported win requires the lower bound > 1. "
                "Five pairs do not establish reliable p95 latency."
            )
            summary = gr.Dataframe(
                headers=SUMMARY_HEADERS,
                datatype=["number"] * 7 + ["bool"],
                interactive=False,
                label="Paired result summary",
            )
            with gr.Tab("First-token timeline"):
                timeline = gr.Plot()
            with gr.Tab("State transfer"):
                transfer_plot = gr.Plot()
            with gr.Tab("Memory and environment"):
                gr.Markdown(
                    "Memory shows before/after snapshots and sampled maxima/minima when reported. "
                    "Inspect the sampling period and sample count in the full report. "
                    "Sampling may miss short transients; unavailable measurements stay empty."
                )
                memory = gr.Dataframe(
                    headers=["Prompt tokens", "Pair", "Mode", "Memory measurements"],
                    interactive=False,
                    wrap=True,
                )
                memory_plot = gr.Plot(label="Memory snapshots")
                details = gr.JSON(label="Full measured report and environment")
            downloads = gr.File(
                label="Download JSON · CSV · standalone HTML", file_count="multiple"
            )
        with gr.Tab("History and comparison"):
            history = gr.Dropdown(choices=history_choices(), multiselect=True, label="Saved runs")
            with gr.Row():
                refresh_button = gr.Button("Refresh history")
                compare_button = gr.Button("Compare selected runs")
            history_plot = gr.Plot(label="Saved run comparison")
            history_downloads = gr.File(file_count="multiple", label="Selected report exports")
        with gr.Tab("Chat"):
            gr.Markdown(
                "Chat uses the same model configuration and single-request runtime as the "
                "benchmark. Previous messages are re-evaluated with an empty prompt cache."
            )
            chat_mode = gr.Radio(["local", "split"], value="split", label="Execution")
            chatbot = gr.Chatbot(label="Conversation")
            prompt = gr.Textbox(label="Message", lines=3)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                chat_stop = gr.Button("Cancel active request", variant="stop")
            chat_status = gr.Textbox(label="Stage / measured metrics", interactive=False, lines=4)
        run.click(
            benchmark,
            [model, cache, context, batch, ubatch, sizes, output, repeats, label],
            [
                status,
                cards,
                chart,
                timeline,
                transfer_plot,
                summary,
                memory,
                memory_plot,
                details,
                downloads,
                history,
                run,
            ],
            concurrency_id="demo-runtime",
            concurrency_limit=1,
        )
        stop.click(cancel, outputs=status, queue=False)
        chat_stop.click(cancel, outputs=chat_status, queue=False)
        send.click(
            chat,
            [prompt, chatbot, chat_mode, model, cache, context, batch, ubatch, output],
            [chatbot, chat_status],
            concurrency_id="demo-runtime",
            concurrency_limit=1,
        )
        refresh_button.click(refresh, outputs=history, queue=False)
        compare_button.click(compare, history, [history_plot, history_downloads], queue=False)
    return demo.queue(default_concurrency_limit=1)
