import csv
import html
import json
import os
from pathlib import Path

from .models import BenchmarkReport


def save_report(report: BenchmarkReport, directory: Path) -> dict[str, str]:
    """Atomic JSON is the history source; CSV retains every trial and runtime timing."""
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / report.run_id
    report.exports = {
        ext: str(base.with_suffix(f".{ext}").resolve()) for ext in ("json", "csv", "html")
    }
    csv_path = Path(report.exports["csv"])
    rows = []
    for sample in report.samples:
        result = sample.result
        row = {
            "benchmark_id": report.run_id,
            "status": report.status,
            "prompt_tokens": sample.prompt_tokens,
            "repeat": sample.repeat + 1,
            "order": sample.order + 1,
            "mode": result.mode,
            "model_id": result.model_id,
            "generated_tokens": result.generated_tokens,
            "success": result.success,
            "error": result.error,
        }
        row.update(result.metrics.model_dump())
        for name in ("memory",):
            row[name] = json.dumps(row[name], ensure_ascii=False, default=str)
        for name in ("fingerprint", "local_profile", "remote_profile"):
            item = getattr(result, name)
            row[name] = item.model_dump_json() if item else ""
        rows.append(row)
    with csv_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            list(rows[0]) if rows else ["benchmark_id", "status", "prompt_tokens", "repeat", "mode"]
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_path.with_suffix(".csv.tmp"), csv_path)
    from .charts import curves, memory_chart, phases, transfer

    figures = [curves([report]), phases(report), transfer(report), memory_chart(report)]
    from plotly.io import to_html

    chart_html = "\n".join(
        to_html(figure, full_html=False, include_plotlyjs=index == 0)
        for index, figure in enumerate(figures)
    )
    escaped = html.escape(report.model_dump_json(indent=2))
    title = html.escape(report.config.label or f"Benchmark {report.run_id[:8]}")
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
    skipped = "".join(
        f"<li>{s.prompt_tokens:,}: {html.escape(s.reason)}</li>" for s in report.skipped
    )
    errors = "".join(f"<li>{html.escape(error)}</li>" for error in report.errors)
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:1rem}}</style></head>
<body><h1>{title}</h1><p>Status: {report.status} · {report.created_at.isoformat()}</p>
<ul>{notes}</ul>{chart_html}<h2>Skipped points</h2><ul>{skipped}</ul>
<h2>Failures</h2><ul>{errors}</ul><details><summary>Full report, environment and memory</summary>
<pre>{escaped}</pre></details></body></html>"""
    html_path = Path(report.exports["html"])
    html_path.with_suffix(".html.tmp").write_text(document, encoding="utf-8")
    os.replace(html_path.with_suffix(".html.tmp"), html_path)
    json_path = Path(report.exports["json"])
    json_path.with_suffix(".json.tmp").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    os.replace(json_path.with_suffix(".json.tmp"), json_path)
    return report.exports


def load_report(path: Path) -> BenchmarkReport:
    return BenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))


def list_reports(directory: Path) -> list[BenchmarkReport]:
    reports = []
    for path in directory.glob("*.json"):
        try:
            reports.append(load_report(path))
        except (ValueError, OSError):
            continue
    return sorted(reports, key=lambda report: report.created_at, reverse=True)
