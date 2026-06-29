import asyncio
import unittest

import numpy as np

from cuebee_speaker.assignment import OnlineSpeakerAssigner
from cuebee_speaker.audio import StreamingAudioFrontend, float32_to_pcm16le
from cuebee_speaker.config import AssignmentConfig, AudioConfig, BatcherConfig
from cuebee_speaker.domain import AudioChunk, STTSegment, STTWord
from cuebee_speaker.gateway import EventGateway
from cuebee_speaker.inference import CrossSessionMicroBatcher
from cuebee_speaker.metrics import MetricRegistry
from cuebee_speaker.pipeline import DiarizationPipeline
from cuebee_speaker.state import InMemorySpeakerStore


class SpectralPeakBackend:
    embedding_dimension = 192

    def infer_batch(self, features):
        embeddings = np.zeros((len(features), self.embedding_dimension), dtype=np.float32)
        for index, item in enumerate(features):
            spectral_peak = int(np.argmax(np.std(item, axis=0)))
            embeddings[index, spectral_peak] = 1.0
        return embeddings, np.ones((len(features),), dtype=np.float32)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        audio_config = AudioConfig(window_ms=1_000, hop_ms=1_000)
        batcher_config = BatcherConfig(max_batch_size=2, max_wait_ms=5)
        self.metrics = MetricRegistry()
        self.pipeline = DiarizationPipeline(
            gateway=EventGateway(),
            frontend=StreamingAudioFrontend(audio_config),
            batcher=CrossSessionMicroBatcher(SpectralPeakBackend(), batcher_config),
            assigner=OnlineSpeakerAssigner(
                InMemorySpeakerStore(), AssignmentConfig(cosine_threshold=0.95)
            ),
            metrics=self.metrics,
        )

    async def asyncTearDown(self) -> None:
        await self.pipeline.close()

    async def _send_second(self, frequency: float, starting_sequence: int) -> list:
        found = []
        for offset in range(10):
            sequence = starting_sequence + offset
            start_ms = sequence * 100
            indices = np.arange(1_600, dtype=np.float32) + start_ms * 16
            time_values = indices / 16_000
            samples = (
                0.22
                * (1.0 + 0.35 * np.sin(2 * np.pi * 3 * time_values))
                * np.sin(2 * np.pi * frequency * time_values)
            ).astype(np.float32)
            result = await self.pipeline.ingest_audio(
                AudioChunk(
                    tenant_id="tenant",
                    session_id="session",
                    client_epoch=1,
                    seq_no=sequence,
                    start_ms=start_ms,
                    end_ms=start_ms + 100,
                    pcm=float32_to_pcm16le(samples),
                )
            )
            found.extend(result.speaker_segments)
        return found

    async def test_end_to_end_audio_assignment_and_word_alignment(self) -> None:
        first = await self._send_second(180, 0)
        second = await self._send_second(380, 10)
        self.assertEqual(first[0].speaker_id, "spk_001")
        self.assertEqual(second[0].speaker_id, "spk_002")

        event = STTSegment(
            tenant_id="tenant",
            session_id="session",
            segment_id="stt-1",
            revision=1,
            text="alpha speaks beta answers",
            start_ms=0,
            end_ms=2_000,
            is_final=True,
            words=(
                STTWord("alpha", 0, 400),
                STTWord("speaks", 400, 900),
                STTWord("beta", 1_050, 1_400),
                STTWord("answers", 1_400, 1_900),
            ),
        )
        spans = self.pipeline.align_stt(event)
        self.assertEqual([item.speaker_id for item in spans], ["spk_001", "spk_002"])
        self.assertEqual(self.metrics.snapshot()["counters"]["speaker_segments"], 2.0)

    async def test_duplicate_packet_does_not_repeat_inference(self) -> None:
        samples = np.zeros((1_600,), dtype=np.float32)
        packet = AudioChunk(
            "tenant", "session", 1, 0, 0, 100, float32_to_pcm16le(samples)
        )
        first = await self.pipeline.ingest_audio(packet)
        duplicate = await self.pipeline.ingest_audio(packet)
        self.assertTrue(first.gateway.accepted)
        self.assertFalse(duplicate.gateway.accepted)
        self.assertEqual(duplicate.speaker_segments, [])


if __name__ == "__main__":
    unittest.main()

