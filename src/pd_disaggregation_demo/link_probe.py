"""Bounded SSH application throughput; all wall timings include SSH setup.

Only Python 3 and SSH are needed on the two hosts. No persistent remote files,
listeners, or processes are installed. Reported rates include encryption, Python
streaming, and SHA-256 verification; they are not raw physical-link capacity.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import selectors
import shlex
import signal
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK_BYTES = 256 * 1024
DEFAULT_BYTES = 64 * 1024 * 1024
METADATA_LIMIT = 64 * 1024
SEED = b"pd-disaggregation-demo/link-probe/v1"
SSH_OPTIONS = [
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "Compression=no",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ControlPersist=no",
    "-o",
    "ForwardAgent=no",
    "-o",
    "ClearAllForwardings=yes",
]

REMOTE_SCRIPT = r"""
import hashlib, json, sys, time
direction, size, chunk_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
block = hashlib.shake_256(b"pd-disaggregation-demo/link-probe/v1").digest(chunk_size)
digest = hashlib.sha256()
count = 0
started = time.monotonic()
if direction == "download":
    stream = sys.stdout.buffer
    while count < size:
        chunk = block[:min(chunk_size, size - count)]
        stream.write(chunk)
        digest.update(chunk)
        count += len(chunk)
    stream.flush()
    metadata_stream = sys.stderr
else:
    stream = sys.stdin.buffer
    while count < size:
        chunk = stream.read(min(chunk_size, size - count))
        if not chunk:
            break
        digest.update(chunk)
        count += len(chunk)
    metadata_stream = sys.stdout
metadata = {"bytes": count, "sha256": digest.hexdigest(),
            "stream_duration_s": time.monotonic() - started,
            "python_version": sys.version.split()[0]}
print("PD_LINK " + json.dumps(metadata), file=metadata_stream, flush=True)
sys.exit(0 if count == size else 2)
"""


def payload_chunks(size: int):
    """Yield at most one bounded pseudorandom block at a time."""
    block = hashlib.shake_256(SEED).digest(CHUNK_BYTES)
    for offset in range(0, size, CHUNK_BYTES):
        yield block[: min(CHUNK_BYTES, size - offset)]


def expected_sha256(size: int) -> str:
    digest = hashlib.sha256()
    for chunk in payload_chunks(size):
        digest.update(chunk)
    return digest.hexdigest()


def rates(byte_count: int, duration: float) -> dict[str, float]:
    return {
        "Mbps": byte_count * 8 / duration / 1_000_000,
        "MBps": byte_count / duration / 1_000_000,
    }


def ssh_command(host: str, command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, "--", host, command]


def remote_command(direction: str, size: int) -> str:
    return "python3 -c " + shlex.quote(REMOTE_SCRIPT) + f" {direction} {size} {CHUNK_BYTES}"


def _terminate(process: subprocess.Popen) -> None:
    """Close the SSH process group, escalating after a short grace period."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1)


def _metadata(raw: bytearray) -> dict[str, Any] | None:
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("PD_LINK "):
            try:
                value = json.loads(line[8:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    return None


def measure_transfer(
    host: str, direction: str, size: int, timeout: float, expected_digest: str
) -> dict[str, Any]:
    """Stream all pipes concurrently so transfer, metadata, and timeout stay bounded."""
    started = time.monotonic()
    result: dict[str, Any] = {
        "direction": direction,
        "requested_bytes": size,
        "local_bytes": 0,
        "remote_bytes": None,
        "status": "failed",
    }
    process = None
    digest = hashlib.sha256()
    stdout, stderr = bytearray(), bytearray()
    block = hashlib.shake_256(SEED).digest(CHUNK_BYTES)
    sent = received = 0
    try:
        process = subprocess.Popen(
            ssh_command(host, remote_command(direction, size)),
            stdin=subprocess.PIPE if direction == "upload" else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            for stream, name in [(process.stdout, "stdout"), (process.stderr, "stderr")]:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            if process.stdin is not None:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            while selector.get_map() or process.poll() is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    result["status"] = "timeout"
                    break
                for key, _ in selector.select(min(0.1, remaining)):
                    stream = key.fileobj
                    fd = key.fd
                    if key.data == "stdin":
                        assert process.stdin is not None
                        try:
                            offset = sent % CHUNK_BYTES
                            count = os.write(
                                fd, block[offset : offset + min(CHUNK_BYTES - offset, size - sent)]
                            )
                            sent += count
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            selector.unregister(stream)
                            process.stdin.close()
                            continue
                        if sent == size:
                            selector.unregister(stream)
                            process.stdin.close()
                    else:
                        try:
                            chunk = os.read(fd, CHUNK_BYTES)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(stream)
                            continue
                        if key.data == "stdout" and direction == "download":
                            digest.update(chunk)
                            received += len(chunk)
                            if received > size:
                                result["status"] = "byte_mismatch"
                                break
                        else:
                            target = stdout if key.data == "stdout" else stderr
                            target.extend(chunk[: max(0, METADATA_LIMIT - len(target))])
                if result["status"] == "byte_mismatch":
                    break
            if result["status"] not in {"timeout", "byte_mismatch"}:
                result["status"] = "completed" if process.returncode == 0 else "ssh_failed"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except OSError as exc:
        result["status"] = "local_io_error"
        result["error_type"] = type(exc).__name__
    finally:
        if process is not None:
            _terminate(process)
            result["ssh_returncode"] = process.returncode
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
    duration = time.monotonic() - started
    result.update(duration_s=duration, local_bytes=sent if direction == "upload" else received)
    metadata = _metadata(stderr if direction == "download" else stdout)
    if metadata is not None:
        result["remote_bytes"] = metadata.get("bytes")
        result["remote_sha256"] = metadata.get("sha256")
        result["remote_stream_duration_s"] = metadata.get("stream_duration_s")
        result["remote_python_version"] = metadata.get("python_version")
    if direction == "download":
        result["local_sha256"] = digest.hexdigest()
    if result["status"] == "completed":
        if result["local_bytes"] != size or result["remote_bytes"] != size:
            result["status"] = "byte_mismatch"
        elif result.get("remote_sha256") != expected_digest or (
            direction == "download" and result["local_sha256"] != expected_digest
        ):
            result["status"] = "checksum_mismatch"
    result["verified"] = result["status"] == "completed"
    result.update(rates(result["local_bytes"], duration))
    return result


def measure_latency(host: str, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {"status": "completed"}
    process = None
    try:
        process = subprocess.Popen(
            ssh_command(host, ":"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        result["ssh_returncode"] = process.wait(timeout=timeout)
        if process.returncode:
            result["status"] = "ssh_failed"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except OSError as exc:
        result.update(status="local_io_error", error_type=type(exc).__name__)
    finally:
        if process is not None:
            _terminate(process)
            result["ssh_returncode"] = process.returncode
    result["duration_s"] = time.monotonic() - started
    return result


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def run_probe(host: str, size: int, repeats: int, timeout: float, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "host_alias": host,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "parameters": {
            "bytes_per_transfer": size,
            "repeats": repeats,
            "timeout_per_command_s": timeout,
            "chunk_bytes": CHUNK_BYTES,
        },
        "provenance": {
            "transport": "SSH with fresh connection per command",
            "ssh_options": SSH_OPTIONS,
            "local_system": platform.system(),
            "local_python_version": platform.python_version(),
            "payload": "repeated deterministic SHAKE-256 pseudorandom block",
            "verification": "exact byte counts and SHA-256 at both endpoints",
            "rate_units": "decimal Mbps and MB/s",
            "duration_scope": "local monotonic wall time including SSH setup and verification",
            "latency_scope": "SSH command wall latency including setup; not ICMP RTT",
            "remote_duration_scope": "Python stream loop including SHA-256; excludes SSH setup",
            "local_byte_scope": "download: read locally; upload: accepted by local SSH stdin",
        },
        "latency_samples": [],
        "transfers": [],
    }
    save_report(output, report)
    expected_digest = expected_sha256(size)
    report["expected_sha256"] = expected_digest
    interrupted = False
    for index in range(3):
        sample = measure_latency(host, timeout)
        report["latency_samples"].append({"sample": index + 1, **sample})
        save_report(output, report)
        if sample["status"] == "interrupted":
            interrupted = True
            break
    if not interrupted:
        for index in range(repeats):
            for direction in ("download", "upload"):
                sample = measure_transfer(host, direction, size, timeout, expected_digest)
                report["transfers"].append({"repeat": index + 1, **sample})
                save_report(output, report)
                if sample["status"] == "interrupted":
                    interrupted = True
                    break
            if interrupted:
                break
    report["summary"] = {}
    latencies = [
        sample["duration_s"] * 1000
        for sample in report["latency_samples"]
        if sample["status"] == "completed"
    ]
    report["summary"]["ssh_command_latency"] = {
        "successful_samples": len(latencies),
        "median_ms": statistics.median(latencies) if latencies else None,
    }
    for direction in ("download", "upload"):
        values = [
            sample["Mbps"]
            for sample in report["transfers"]
            if sample["direction"] == direction and sample["verified"]
        ]
        report["summary"][direction] = {
            "successful_samples": len(values),
            "median_Mbps": statistics.median(values) if values else None,
            "median_MBps": statistics.median(values) / 8 if values else None,
        }
    samples = report["latency_samples"] + report["transfers"]
    report["status"] = (
        "interrupted"
        if interrupted
        else (
            "completed" if all(sample["status"] == "completed" for sample in samples) else "partial"
        )
    )
    report["finished_at"] = datetime.now(UTC).isoformat()
    save_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="gpu_home", help="SSH host alias (default: gpu_home)")
    parser.add_argument(
        "--bytes",
        type=int,
        default=DEFAULT_BYTES,
        help="payload bytes per direction and repeat (default: 67108864)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120, help="seconds per SSH command")
    parser.add_argument("--output", type=Path, required=True, help="destination JSON report")
    args = parser.parse_args()
    if args.bytes <= 0 or args.repeats <= 0 or args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("bytes, repeats, and finite timeout must be positive")
    if not args.host or args.host.startswith("-") or any(c.isspace() for c in args.host):
        parser.error("host must be a single SSH host alias")
    report = run_probe(args.host, args.bytes, args.repeats, args.timeout, args.output)
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output), "summary": report["summary"]},
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
