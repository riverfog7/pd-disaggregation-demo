import hashlib
import json
import sys

import pytest

from pd_disaggregation_demo import link_probe


def local_remote(monkeypatch, source=link_probe.REMOTE_SCRIPT):
    """Run the exact remote stream code locally without any SSH/network access."""
    monkeypatch.setattr(
        link_probe,
        "ssh_command",
        lambda host, command: [sys.executable, "-c", source, *command.rsplit(" ", 3)[-3:]],
    )


def test_rates_use_decimal_bits_and_bytes():
    assert link_probe.rates(64 * 1024 * 1024, 2) == {
        "Mbps": 268.435456,
        "MBps": 33.554432,
    }


@pytest.mark.parametrize("size", [1, link_probe.CHUNK_BYTES, 2 * link_probe.CHUNK_BYTES + 13])
def test_payload_chunks_are_bounded_and_exact(size):
    chunks = list(link_probe.payload_chunks(size))
    assert sum(map(len, chunks)) == size
    assert all(0 < len(chunk) <= link_probe.CHUNK_BYTES for chunk in chunks)
    assert hashlib.sha256(b"".join(chunks)).hexdigest() == link_probe.expected_sha256(size)
    if len(chunks) > 1:
        assert chunks[1] == chunks[0][: len(chunks[1])]


@pytest.mark.parametrize("direction", ["download", "upload"])
@pytest.mark.parametrize("size", [1, 2 * link_probe.CHUNK_BYTES + 13])
def test_real_stream_code_counts_and_verifies_both_directions(monkeypatch, direction, size):
    local_remote(monkeypatch)
    result = link_probe.measure_transfer(
        "unused", direction, size, 5, link_probe.expected_sha256(size)
    )
    assert result["status"] == "completed"
    assert result["verified"]
    assert result["local_bytes"] == result["remote_bytes"] == size
    assert result["remote_sha256"] == link_probe.expected_sha256(size)
    assert result["duration_s"] > result["remote_stream_duration_s"] > 0


def test_download_checksum_mismatch_is_excluded(monkeypatch):
    source = link_probe.REMOTE_SCRIPT.replace("link-probe/v1", "link-probe/v2")
    local_remote(monkeypatch, source)
    result = link_probe.measure_transfer(
        "unused", "download", 37, 5, link_probe.expected_sha256(37)
    )
    assert result["status"] == "checksum_mismatch"
    assert result["local_bytes"] == result["remote_bytes"] == 37
    assert not result["verified"]


def test_remote_count_mismatch_is_not_success(monkeypatch):
    source = link_probe.REMOTE_SCRIPT.replace('"bytes": count', '"bytes": count - 1')
    local_remote(monkeypatch, source)
    result = link_probe.measure_transfer("unused", "upload", 37, 5, link_probe.expected_sha256(37))
    assert result["status"] == "byte_mismatch"
    assert result["local_bytes"] == 37 and result["remote_bytes"] == 36
    assert not result["verified"]


def test_timeout_keeps_partial_download_bytes_and_reaps_process(monkeypatch):
    size = 2 * link_probe.CHUNK_BYTES
    source = (
        "import sys,time; sys.stdout.buffer.write(b'x'*4096); sys.stdout.flush(); time.sleep(10)"
    )
    local_remote(monkeypatch, source)
    processes = []
    original = link_probe.subprocess.Popen

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(link_probe.subprocess, "Popen", capture)
    result = link_probe.measure_transfer(
        "unused", "download", size, 0.2, link_probe.expected_sha256(size)
    )
    assert result["status"] == "timeout"
    assert result["local_bytes"] == 4096 and result["remote_bytes"] is None
    assert not result["verified"]
    assert result["duration_s"] < 2
    assert processes[0].poll() is not None


def test_timeout_keeps_partial_upload_bytes(monkeypatch):
    local_remote(monkeypatch, "import time; time.sleep(10)")
    size = 2 * link_probe.CHUNK_BYTES
    result = link_probe.measure_transfer(
        "unused", "upload", size, 0.2, link_probe.expected_sha256(size)
    )
    assert result["status"] == "timeout"
    assert 0 < result["local_bytes"] < size
    assert result["remote_bytes"] is None


def test_probe_alternates_and_saves_partial_result_after_each_sample(monkeypatch, tmp_path):
    saved, order = [], []
    original_save = link_probe.save_report

    def save(path, report):
        original_save(path, report)
        saved.append(json.loads(path.read_text()))

    def transfer(host, direction, size, timeout, expected_digest):
        order.append(direction)
        failed = len(order) == 2
        return {
            "direction": direction,
            "status": "timeout" if failed else "completed",
            "verified": not failed,
            "Mbps": 100 if failed else 8,
            "local_bytes": size,
        }

    monkeypatch.setattr(link_probe, "save_report", save)
    monkeypatch.setattr(
        link_probe, "measure_latency", lambda *args: {"status": "completed", "duration_s": 0.1}
    )
    monkeypatch.setattr(link_probe, "measure_transfer", transfer)
    output = tmp_path / "nested" / "link.json"
    report = link_probe.run_probe("gpu_home", 17, 2, 1, output)
    assert order == ["download", "upload", "download", "upload"]
    assert [len(value["transfers"]) for value in saved] == [0, 0, 0, 0, 1, 2, 3, 4, 4]
    assert report["status"] == "partial"
    assert report["summary"]["upload"] == {
        "successful_samples": 1,
        "median_Mbps": 8,
        "median_MBps": 1,
    }
    assert json.loads(output.read_text()) == report


def test_ssh_options_disable_compression_and_connection_reuse():
    command = link_probe.ssh_command("gpu_home", ":")
    for option in (
        "Compression=no",
        "ControlMaster=no",
        "ControlPath=none",
        "ControlPersist=no",
        "BatchMode=yes",
        "ConnectTimeout=10",
    ):
        assert option in command
    assert command[-3:] == ["--", "gpu_home", ":"]
