"""Small dependency-free metric registry with Prometheus-compatible export."""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, Mapping

import numpy as np

from .inference.batcher import BatchObservation


class MetricRegistry:
    def __init__(self, max_observations: int = 8_192) -> None:
        self._max_observations = max_observations
        self._counters: DefaultDict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, value: float = 1.0) -> None:
        if value < 0:
            raise ValueError("counter increments cannot be negative")
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            return
        with self._lock:
            values = self._histograms.get(name)
            if values is None:
                values = deque(maxlen=self._max_observations)
                self._histograms[name] = values
            values.append(number)

    def snapshot(self) -> Mapping[str, object]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {
                name: self._summarize(list(values)) for name, values in self._histograms.items()
            }
        return {"counters": counters, "gauges": gauges, "histograms": histograms}

    def prometheus_text(self, prefix: str = "cuebee_speaker") -> str:
        snapshot = self.snapshot()
        lines = []
        for name, value in sorted(snapshot["counters"].items()):  # type: ignore
            lines.append(f"{prefix}_{self._safe_name(name)}_total {value}")
        for name, value in sorted(snapshot["gauges"].items()):  # type: ignore
            lines.append(f"{prefix}_{self._safe_name(name)} {value}")
        for name, summary in sorted(snapshot["histograms"].items()):  # type: ignore
            safe_name = self._safe_name(name)
            for quantile, label in (("p50", "0.50"), ("p95", "0.95"), ("p99", "0.99")):
                lines.append(
                    f'{prefix}_{safe_name}{{quantile="{label}"}} {summary[quantile]}'
                )
            lines.append(f"{prefix}_{safe_name}_count {summary['count']}")
            lines.append(f"{prefix}_{safe_name}_sum {summary['sum']}")
        return "\n".join(lines) + "\n"

    def observe_batch(self, observation: BatchObservation) -> None:
        self.increment("batches")
        self.increment("embedded_audio_seconds", observation.audio_seconds)
        self.observe("batch_size", observation.batch_size)
        self.observe("queue_delay_ms", observation.oldest_queue_delay_ms)
        self.observe("inference_latency_ms", observation.inference_ms)

    @staticmethod
    def _summarize(values: list) -> Mapping[str, float]:
        if not values:
            return {"count": 0, "sum": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "sum": float(np.sum(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
        }

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in name)
