"""Speaker embedding backends and cross-session scheduling."""

from .backend import DeterministicEmbeddingBackend, EmbeddingBackend
from .onnx_backend import ONNXEmbeddingBackend

__all__ = ["DeterministicEmbeddingBackend", "EmbeddingBackend", "ONNXEmbeddingBackend"]

