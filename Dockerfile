ARG TARGETARCH

FROM debian:bookworm-slim AS native-builder

ARG TARGETARCH
ARG ONNXRUNTIME_VERSION=1.27.0

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates cmake curl g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl --fail --location --silent --show-error \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
      --output /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1

WORKDIR /source
COPY CMakeLists.txt ./
COPY cpp ./cpp
RUN cmake -S . -B build \
      -DCUEBEE_ENABLE_ONNX=ON \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY --from=native-builder /source/build/cpp/cuebee-speaker-worker /usr/local/bin/
COPY --from=native-builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
RUN pip install --no-cache-dir '.[onnx]' \
    && useradd --create-home --uid 10001 cuebee \
    && mkdir -p /app/runtime \
    && chown -R cuebee:cuebee /app/runtime

USER cuebee
EXPOSE 8765

CMD ["cuebee-speaker-server", "--host", "0.0.0.0", "--port", "8765", "--native-worker", "/usr/local/bin/cuebee-speaker-worker", "--native-backend", "deterministic"]
