import asyncio
import threading
import time
import unittest

import numpy as np

from cuebee_speaker.config import BatcherConfig
from cuebee_speaker.domain import FeatureChunk
from cuebee_speaker.inference import CrossSessionMicroBatcher


class RecordingBackend:
    embedding_dimension = 192

    def __init__(self) -> None:
        self.batch_shapes = []
        self._lock = threading.Lock()

    def infer_batch(self, features):
        time.sleep(0.002)
        with self._lock:
            self.batch_shapes.append(tuple(item.shape[0] for item in features))
        output = np.zeros((len(features), self.embedding_dimension), dtype=np.float32)
        for index, item in enumerate(features):
            output[index, index % self.embedding_dimension] = 1.0
        return output, np.ones((len(features),), dtype=np.float32)


def feature(session: str, frames: int = 150) -> FeatureChunk:
    return FeatureChunk(
        tenant_id="tenant",
        session_id=session,
        chunk_id=f"{session}-{frames}",
        start_ms=0,
        end_ms=1_500,
        features=np.ones((frames, 80), dtype=np.float32),
        speech_ratio=1.0,
    )


class BatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_session_requests_fill_one_batch(self) -> None:
        backend = RecordingBackend()
        batcher = CrossSessionMicroBatcher(
            backend,
            BatcherConfig(max_batch_size=4, max_wait_ms=50, per_session_batch_limit=1),
        )
        results = await asyncio.gather(
            *(batcher.submit(feature(f"session-{index}")) for index in range(4))
        )
        await batcher.close()
        self.assertEqual([item.batch_size for item in results], [4, 4, 4, 4])
        self.assertEqual(backend.batch_shapes, [(150, 150, 150, 150)])

    async def test_length_buckets_isolate_long_request(self) -> None:
        backend = RecordingBackend()
        batcher = CrossSessionMicroBatcher(
            backend,
            BatcherConfig(max_batch_size=4, max_wait_ms=5, bucket_boundaries_frames=(200,)),
        )
        results = await asyncio.gather(
            batcher.submit(feature("short", 150)),
            batcher.submit(feature("long", 350)),
        )
        await batcher.close()
        self.assertEqual(sorted(item.batch_size for item in results), [1, 1])
        self.assertEqual({shape for batch in backend.batch_shapes for shape in batch}, {150, 350})

    async def test_backlog_reports_audio_seconds(self) -> None:
        backend = RecordingBackend()
        batcher = CrossSessionMicroBatcher(
            backend,
            BatcherConfig(max_batch_size=8, max_wait_ms=100),
        )
        task = asyncio.create_task(batcher.submit(feature("session")))
        await asyncio.sleep(0)
        count, audio_seconds = await batcher.backlog()
        self.assertEqual(count, 1)
        self.assertAlmostEqual(audio_seconds, 1.5)
        result = await task
        self.assertLessEqual(result.queue_delay_ms, 150)
        await batcher.close()


if __name__ == "__main__":
    unittest.main()

