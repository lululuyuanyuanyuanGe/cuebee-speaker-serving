"""Speaker embedding backends and cross-session scheduling."""

from .backend import DeterministicEmbeddingBackend, EmbeddingBackend
from .batcher import BatchObservation, CrossSessionMicroBatcher
from .onnx_backend import ONNXEmbeddingBackend

__all__ = [
    "BatchObservation",
    "CrossSessionMicroBatcher",
    "DeterministicEmbeddingBackend",
    "EmbeddingBackend",
    "ONNXEmbeddingBackend",
]
