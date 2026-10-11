# CUDA ABI and engine revision are deliberately fixed across Metal/CUDA workers.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS engine
ARG ENGINE_REVISION=23b0202a189c44a54625aadcb37a946dd1d6278d
ARG BUILD_JOBS=1
RUN test "$ENGINE_REVISION" = 23b0202a189c44a54625aadcb37a946dd1d6278d \
    && test "$BUILD_JOBS" -ge 1 && test "$BUILD_JOBS" -le 2
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build g++ ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git init /src/llama.cpp && cd /src/llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "$ENGINE_REVISION" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$ENGINE_REVISION"
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build /src/llama.cpp/build --target llama-server --parallel "$BUILD_JOBS" \
    && mkdir -p /opt/llama/bin /opt/llama/lib \
    && cp /src/llama.cpp/build/bin/llama-server /opt/llama/bin/ \
    && find /src/llama.cpp/build -name '*.so*' \( -type f -o -type l \) -exec cp -P {} /opt/llama/lib/ \;

FROM ghcr.io/astral-sh/uv:0.9.18 AS uv
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS python
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
RUN uv python install 3.13
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --python 3.13 --no-editable

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime
ARG GIT_REVISION=unknown
LABEL org.opencontainers.image.revision=$GIT_REVISION \
      org.pd-demo.engine.revision=23b0202a189c44a54625aadcb37a946dd1d6278d
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/* \
    && if ! getent passwd 1000 >/dev/null; then useradd --create-home --uid 1000 worker; fi \
    && install -d -o 1000 -g 1000 /home/worker
COPY --from=engine /opt/llama /opt/llama
COPY --from=python /opt/python /opt/python
COPY --from=python /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PDD_ENGINE_BINARY=/opt/llama/bin/llama-server \
    PDD_CACHE_DIR=/data/cache PDD_WORK_DIR=/data/work \
    PDD_HOST=0.0.0.0 PDD_WORKER_PORT=8090 PDD_GIT_REVISION=$GIT_REVISION \
    PYTHONUNBUFFERED=1 HOME=/home/worker HF_HUB_DISABLE_XET=1
USER 1000:1000
WORKDIR /app
EXPOSE 8090
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4)" || exit 1
ENTRYPOINT ["pd-demo", "worker"]
