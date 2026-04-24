"""ONNX Runtime backend for the 3D-Speaker ERes2Net embedding artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .backend import normalize_rows


class ONNXEmbeddingBackend:
    """Execute a `[batch, time, 80] -> [batch, 192]` speaker model."""

    embedding_dimension = 192

    def __init__(
        self,
        model_path: str,
        providers: Optional[Sequence[str]] = None,
        intra_op_threads: int = 0,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment-dependent branch
            raise RuntimeError("install the project with the 'onnx' extra") from error

        options = ort.SessionOptions()
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        selected_providers = list(providers or ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=selected_providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def infer_batch(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not features:
            return (
                np.empty((0, self.embedding_dimension), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        mel_bins = {int(item.shape[1]) for item in features if item.ndim == 2}
        if mel_bins != {80} or len(mel_bins) != 1:
            raise ValueError("the ERes2Net artifact expects [time, 80] feature sequences")
        maximum_frames = max(item.shape[0] for item in features)
        batch = np.zeros((len(features), maximum_frames, 80), dtype=np.float32)
        for index, item in enumerate(features):
            batch[index, : item.shape[0]] = item
        output = self._session.run([self._output_name], {self._input_name: batch})[0]
        embeddings = normalize_rows(np.asarray(output, dtype=np.float32).reshape(len(batch), -1))
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"unexpected embedding dimension: {embeddings.shape[1]}")
        duration_quality = np.asarray(
            [min(1.0, item.shape[0] / 100.0) for item in features], dtype=np.float32
        )
        return embeddings, duration_quality

