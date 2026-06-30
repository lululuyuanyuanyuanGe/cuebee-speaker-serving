"""Speaker embedding backends and cross-session scheduling."""

from .backend import DeterministicEmbeddingBackend, EmbeddingBackend
from .batcher import BatchObservation, CrossSessionMicroBatcher
from .native_worker import NativeWorkerBackend
from .onnx_backend import ONNXEmbeddingBackend

__all__ = [
    "BatchObservation",
    "CrossSessionMicroBatcher",
    "DeterministicEmbeddingBackend",
    "EmbeddingBackend",
    "NativeWorkerBackend",
    "ONNXEmbeddingBackend",
]
