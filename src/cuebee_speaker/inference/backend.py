"""Backend interface plus a deterministic development embedding implementation."""

from __future__ import annotations

import threading
from typing import Protocol, Sequence, Tuple

import numpy as np


class EmbeddingBackend(Protocol):
    embedding_dimension: int

    def infer_batch(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalized embeddings and one quality score per feature sequence."""


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class DeterministicEmbeddingBackend:
    """A deterministic 192-dimensional backend for development and scheduling tests.

    It is not an acoustic model and its output is never an accuracy result. It keeps the
    full serving path runnable when the licensed or separately distributed ONNX artifact
    is unavailable.
    """

    embedding_dimension = 192

    def __init__(self, seed: int = 17) -> None:
        self._seed = seed
        self._projection = None  # type: ignore
        self._lock = threading.Lock()

    def infer_batch(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not features:
            return (
                np.empty((0, self.embedding_dimension), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        statistics = np.stack([self._summarize(item) for item in features])
        with self._lock:
            if self._projection is None:
                generator = np.random.default_rng(self._seed)
                self._projection = generator.standard_normal(
                    (statistics.shape[1], self.embedding_dimension), dtype=np.float32
                ) / np.sqrt(statistics.shape[1])
            projection = self._projection
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            projected = statistics @ projection
        if not np.all(np.isfinite(projected)):
            raise FloatingPointError("development embedding projection produced non-finite values")
        embeddings = normalize_rows(np.tanh(projected))
        qualities = np.asarray([self._quality(item) for item in features], dtype=np.float32)
        return embeddings.astype(np.float32), qualities

    @staticmethod
    def _summarize(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("features must have shape [time, mel] with non-empty time")
        temporal_bins = []
        for indices in np.array_split(np.arange(values.shape[0]), 4):
            temporal_bins.append(np.mean(values[indices], axis=0))
        delta = np.diff(values, axis=0)
        if delta.size == 0:
            delta = np.zeros_like(values[:1])
        return np.concatenate(
            (
                np.mean(values, axis=0),
                np.std(values, axis=0),
                np.mean(np.abs(delta), axis=0),
                *temporal_bins,
            )
        ).astype(np.float32)

    @staticmethod
    def _quality(features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float32)
        duration_factor = min(1.0, values.shape[0] / 100.0)
        finite_ratio = float(np.mean(np.isfinite(values)))
        dynamic_factor = min(1.0, float(np.std(values)) / 2.0)
        return float(np.clip(duration_factor * finite_ratio * dynamic_factor, 0.0, 1.0))
