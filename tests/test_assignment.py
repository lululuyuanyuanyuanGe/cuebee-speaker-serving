import os
import tempfile
import unittest

import numpy as np

from cuebee_speaker.assignment import OnlineSpeakerAssigner
from cuebee_speaker.config import AssignmentConfig
from cuebee_speaker.domain import EmbeddingResult, FeatureChunk
from cuebee_speaker.state import InMemorySpeakerStore, SQLiteSpeakerStore


def result(tenant: str, session: str, vector: np.ndarray, quality: float = 0.9):
    chunk = FeatureChunk(
        tenant_id=tenant,
        session_id=session,
        chunk_id="chunk",
        start_ms=0,
        end_ms=1_500,
        features=np.ones((150, 80), dtype=np.float32),
        speech_ratio=1.0,
    )
    return EmbeddingResult(chunk, vector, quality, "worker", 1, 1.0, 1.0)


class AssignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_assigns_stable_ids_and_creates_new_centroid(self) -> None:
        assigner = OnlineSpeakerAssigner(
            InMemorySpeakerStore(), AssignmentConfig(cosine_threshold=0.8)
        )
        first_vector = np.eye(1, 192, 0, dtype=np.float32)[0]
        second_vector = np.eye(1, 192, 1, dtype=np.float32)[0]
        first = await assigner.assign(result("tenant", "session", first_vector))
        repeated = await assigner.assign(result("tenant", "session", first_vector))
        second = await assigner.assign(result("tenant", "session", second_vector))

        self.assertEqual(first.speaker_id, "spk_001")
        self.assertEqual(repeated.speaker_id, "spk_001")
        self.assertEqual(second.speaker_id, "spk_002")
        await assigner.close()

    async def test_low_quality_embedding_is_not_written(self) -> None:
        assigner = OnlineSpeakerAssigner(
            InMemorySpeakerStore(), AssignmentConfig(min_quality=0.5)
        )
        vector = np.eye(1, 192, 0, dtype=np.float32)[0]
        segment = await assigner.assign(result("tenant", "session", vector, quality=0.1))
        self.assertEqual(segment.speaker_id, "spk_unknown")
        self.assertEqual(await assigner.profiles("tenant", "session"), ())
        await assigner.close()

    async def test_sqlite_recovers_identity_after_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.sqlite3")
            vector = np.eye(1, 192, 7, dtype=np.float32)[0]
            first_assigner = OnlineSpeakerAssigner(
                SQLiteSpeakerStore(path), AssignmentConfig(cosine_threshold=0.8)
            )
            first = await first_assigner.assign(result("tenant", "session", vector))
            await first_assigner.close()

            recovered_assigner = OnlineSpeakerAssigner(
                SQLiteSpeakerStore(path), AssignmentConfig(cosine_threshold=0.8)
            )
            recovered = await recovered_assigner.assign(result("tenant", "session", vector))
            profiles = await recovered_assigner.profiles("tenant", "session")
            self.assertEqual(first.speaker_id, recovered.speaker_id)
            self.assertEqual(profiles[0].sample_count, 2)
            await recovered_assigner.close()

    async def test_tenant_profiles_are_isolated(self) -> None:
        assigner = OnlineSpeakerAssigner(InMemorySpeakerStore(), AssignmentConfig())
        vector = np.eye(1, 192, 0, dtype=np.float32)[0]
        one = await assigner.assign(result("tenant-a", "shared-id", vector))
        two = await assigner.assign(result("tenant-b", "shared-id", vector))
        self.assertEqual(one.speaker_id, "spk_001")
        self.assertEqual(two.speaker_id, "spk_001")
        self.assertEqual(len(await assigner.profiles("tenant-a", "shared-id")), 1)
        self.assertEqual(len(await assigner.profiles("tenant-b", "shared-id")), 1)
        await assigner.close()


if __name__ == "__main__":
    unittest.main()

