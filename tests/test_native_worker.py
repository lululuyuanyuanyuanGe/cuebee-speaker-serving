from __future__ import annotations

import os
import signal
import unittest

import numpy as np

from cuebee_speaker.inference import NativeWorkerBackend


_WORKER = os.environ.get("CUEBEE_NATIVE_WORKER")
_ONNX_MODEL = os.environ.get("CUEBEE_TEST_ONNX_MODEL")


@unittest.skipUnless(_WORKER and os.path.isfile(_WORKER), "native worker was not built")
class NativeWorkerBackendTests(unittest.TestCase):
    def test_deterministic_batch_and_normalization(self) -> None:
        generator = np.random.default_rng(123)
        features = [
            generator.normal(size=(40, 80)).astype(np.float32),
            generator.normal(size=(75, 80)).astype(np.float32),
        ]
        with NativeWorkerBackend(_WORKER or "", timeout_seconds=2.0) as backend:
            embeddings, qualities = backend.infer_batch(features)
        self.assertEqual(embeddings.shape, (2, 192))
        self.assertEqual(qualities.shape, (2,))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all((qualities >= 0.0) & (qualities <= 1.0)))

    def test_restarts_after_unexpected_exit(self) -> None:
        features = [np.ones((25, 80), dtype=np.float32)]
        with NativeWorkerBackend(_WORKER or "", timeout_seconds=2.0) as backend:
            original_pid = backend.worker_pid
            self.assertIsNotNone(original_pid)
            os.kill(int(original_pid or 0), signal.SIGKILL)
            embeddings, _ = backend.infer_batch(features)
            self.assertNotEqual(backend.worker_pid, original_pid)
        self.assertEqual(embeddings.shape, (1, 192))

    def test_rejects_invalid_feature_shape_before_transport(self) -> None:
        with NativeWorkerBackend(_WORKER or "", timeout_seconds=2.0) as backend:
            with self.assertRaisesRegex(ValueError, r"\[time, 80\]"):
                backend.infer_batch([np.ones((20, 79), dtype=np.float32)])

    @unittest.skipUnless(
        _ONNX_MODEL and os.path.isfile(_ONNX_MODEL), "native ONNX test model is unavailable"
    )
    def test_native_onnx_runtime_path(self) -> None:
        generator = np.random.default_rng(456)
        features = [generator.normal(size=(31, 80)).astype(np.float32)]
        with NativeWorkerBackend(
            _WORKER or "",
            model_path=_ONNX_MODEL,
            backend="onnx",
            timeout_seconds=5.0,
        ) as backend:
            embeddings, qualities = backend.infer_batch(features)
        self.assertEqual(embeddings.shape, (1, 192))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
        self.assertGreater(float(qualities[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
