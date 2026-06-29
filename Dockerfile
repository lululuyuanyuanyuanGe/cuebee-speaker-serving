FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[onnx]' \
    && useradd --create-home --uid 10001 cuebee \
    && mkdir -p /app/runtime \
    && chown -R cuebee:cuebee /app/runtime

USER cuebee
EXPOSE 8765

CMD ["cuebee-speaker-server", "--host", "0.0.0.0", "--port", "8765"]
