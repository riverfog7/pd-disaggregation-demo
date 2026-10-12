"""Guard engine cache scope and publication outside its ephemeral mount."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_engine_cache_keeps_upstream_linker_policy_and_publishes_in_same_run():
    dockerfile = (ROOT / "Dockerfile").read_text()
    instructions = re.sub(r"\\\n\s*", " ", dockerfile).splitlines()
    engine = next(line for line in instructions if line.startswith("RUN --mount=type=cache,"))
    options, command = engine.removeprefix("RUN ").split(" ", 1)
    mount = dict(item.split("=", 1) for item in options.removeprefix("--mount=").split(","))

    assert mount["target"] == "/src/llama.cpp/build"
    assert mount["sharing"] == "locked"
    assert "${ENGINE_REVISION}" in mount["id"]
    assert "cuda12.8.1-sm75-ubuntu24.04-release" in mount["id"]
    assert "nativeoff-cudaon-testsoff-serveron-curloff-allowshlib-v1" in mount["id"]
    assert "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined" in command
    assert "-B " + mount["target"] in command
    assert "--target llama-server" in command
    assert "--target llama-server llama-fit-params" in command
    assert "cp /src/llama.cpp/build/bin/llama-server /opt/llama/bin/" in command
    assert "cp /src/llama.cpp/build/bin/llama-fit-params /opt/llama/bin/" in command
    assert "find /src/llama.cpp/build -name '*.so*'" in command
    assert "-exec cp -P {} /opt/llama/lib/" in command
    assert command.index("cmake --build") < command.index("cp /src/llama.cpp/build/bin/")
    assert "COPY --from=engine /opt/llama /opt/llama" in instructions
    assert "/stubs" not in dockerfile
    assert not any("COPY --from=engine" in line and "/build" in line for line in instructions)

    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_server_context_probe_is_built_and_published_alongside_generic_probe():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert (
        "COPY src/pd_disaggregation_demo/runtime/native/server_fit /src/pd-server-fit-params"
        in dockerfile
    )
    assert "-DCMAKE_PROJECT_INCLUDE=/src/pd-server-fit-params/inject.cmake" in dockerfile
    assert "--target llama-server llama-fit-params pd-server-fit-params" in dockerfile
    assert "cp /src/llama.cpp/build/bin/pd-server-fit-params /opt/llama/bin/" in dockerfile
