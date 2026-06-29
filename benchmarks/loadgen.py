#!/usr/bin/env python3
"""Synthetic cross-session batching load generator."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Optional

import numpy as np

from cuebee_speaker.config import BatcherConfig
from cuebee_speaker.domain import FeatureChunk
from cuebee_speaker.inference import (
    CrossSessionMicroBatcher,
    DeterministicEmbeddingBackend,
    ONNXEmbeddingBackend,
)
from cuebee_speaker.metrics import MetricRegistry


def _percentiles(values: list) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


async def run(args: argparse.Namespace) -> dict:
    if args.model:
        backend = ONNXEmbeddingBackend(args.model, intra_op_threads=args.intra_op_threads)
        backend_name = "onnxruntime-eres2net"
    else:
        backend = DeterministicEmbeddingBackend()
        backend_name = "deterministic-development-backend"
    metrics = MetricRegistry()
    batcher = CrossSessionMicroBatcher(
        backend,
        BatcherConfig(
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
            per_session_batch_limit=args.per_session_batch_limit,
        ),
        observer=metrics.observe_batch,
    )
    latencies_ms = []
    queue_ms = []
    inference_ms = []
    batch_sizes = []
    started_at = time.perf_counter()

    async def session_client(session_index: int) -> None:
        generator = np.random.default_rng(args.seed + session_index)
        template = generator.normal(size=(args.frames, 80)).astype(np.float32)
        for request_index in range(args.requests_per_session):
            if args.arrival_interval_ms > 0 and request_index:
                await asyncio.sleep(args.arrival_interval_ms / 1_000.0)
            chunk = FeatureChunk(
                tenant_id=f"tenant-{session_index % args.tenants}",
                session_id=f"session-{session_index}",
                chunk_id=f"{session_index}:{request_index}",
                start_ms=request_index * 750,
                end_ms=request_index * 750 + args.audio_ms,
                features=template,
                speech_ratio=1.0,
            )
            request_started = time.perf_counter()
            result = await batcher.submit(chunk, deadline_ms=args.deadline_ms)
            latencies_ms.append((time.perf_counter() - request_started) * 1_000.0)
            queue_ms.append(result.queue_delay_ms)
            inference_ms.append(result.inference_ms)
            batch_sizes.append(result.batch_size)

    try:
        await asyncio.gather(*(session_client(index) for index in range(args.sessions)))
    finally:
        await batcher.close()
    elapsed = time.perf_counter() - started_at
    requests = args.sessions * args.requests_per_session
    audio_seconds = requests * args.audio_ms / 1_000.0
    return {
        "evidence_level": "LAB",
        "warning": (
            "Synthetic features and scheduling path only; not speaker accuracy or "
            "production end-to-end latency."
        ),
        "backend": backend_name,
        "configuration": {
            "tenants": args.tenants,
            "sessions": args.sessions,
            "requests_per_session": args.requests_per_session,
            "feature_shape": [args.frames, 80],
            "audio_ms_per_request": args.audio_ms,
            "max_batch_size": args.max_batch_size,
            "max_wait_ms": args.max_wait_ms,
            "deadline_ms": args.deadline_ms,
        },
        "results": {
            "requests": requests,
            "elapsed_seconds": elapsed,
            "requests_per_second": requests / elapsed,
            "audio_seconds_per_second": audio_seconds / elapsed,
            "request_latency_ms": _percentiles(latencies_ms),
            "queue_delay_ms": _percentiles(queue_ms),
            "inference_latency_ms": _percentiles(inference_ms),
            "mean_batch_size": float(np.mean(batch_sizes)),
        },
        "metrics": metrics.snapshot(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CueBee speaker-serving synthetic load test")
    parser.add_argument("--tenants", type=int, default=4)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--requests-per-session", type=int, default=10)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--audio-ms", type=int, default=1_500)
    parser.add_argument("--arrival-interval-ms", type=float, default=0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=int, default=80)
    parser.add_argument("--deadline-ms", type=int, default=250)
    parser.add_argument("--per-session-batch-limit", type=int, default=2)
    parser.add_argument("--model", help="optional ERes2Net ONNX model path")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    if min(args.tenants, args.sessions, args.requests_per_session) <= 0:
        raise SystemExit("tenant, session, and request counts must be positive")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()

