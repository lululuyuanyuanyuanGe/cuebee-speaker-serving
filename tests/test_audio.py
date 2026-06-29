import unittest

import numpy as np

from cuebee_speaker.audio import (
    EnergyVAD,
    LogMelFbank,
    StreamingAudioFrontend,
    float32_to_pcm16le,
    pcm16le_to_float32,
)
from cuebee_speaker.config import AudioConfig
from cuebee_speaker.domain import AudioChunk


class AudioFrontendTests(unittest.TestCase):
    def test_pcm_round_trip(self) -> None:
        source = np.asarray([-1.0, -0.5, 0.0, 0.5, 0.999], dtype=np.float32)
        restored = pcm16le_to_float32(float32_to_pcm16le(source))
        np.testing.assert_allclose(restored, source, atol=1.0 / 32_768.0)

    def test_energy_vad_distinguishes_speech_energy_from_silence(self) -> None:
        rate = 16_000
        time = np.arange(rate, dtype=np.float32) / rate
        speech = 0.2 * np.sin(2 * np.pi * 220 * time)
        vad = EnergyVAD(rate, threshold_dbfs=-45)
        self.assertGreater(vad.analyze(speech).speech_ratio, 0.95)
        self.assertEqual(vad.analyze(np.zeros_like(speech)).speech_ratio, 0.0)

    def test_log_mel_shape_and_finiteness(self) -> None:
        rate = 16_000
        time = np.arange(rate * 3 // 2, dtype=np.float32) / rate
        speech = 0.2 * np.sin(2 * np.pi * 220 * time)
        features = LogMelFbank().extract(speech)
        self.assertEqual(features.shape, (149, 80))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_streaming_frontend_emits_overlapping_windows(self) -> None:
        config = AudioConfig(window_ms=1_500, hop_ms=750)
        frontend = StreamingAudioFrontend(config)
        found = []
        for sequence in range(23):
            start_ms = sequence * 100
            duration_ms = 50 if sequence == 22 else 100
            sample_count = config.sample_rate * duration_ms // 1_000
            offset = start_ms * config.sample_rate // 1_000
            indices = np.arange(offset, offset + sample_count, dtype=np.float32)
            samples = 0.2 * np.sin(2 * np.pi * 220 * indices / config.sample_rate)
            found.extend(
                frontend.push(
                    AudioChunk(
                        tenant_id="tenant",
                        session_id="session",
                        client_epoch=1,
                        seq_no=sequence,
                        start_ms=start_ms,
                        end_ms=start_ms + duration_ms,
                        pcm=float32_to_pcm16le(samples),
                    )
                )
            )
        self.assertEqual([(item.start_ms, item.end_ms) for item in found], [(0, 1500), (750, 2250)])
        self.assertTrue(all(item.features.shape[1] == 80 for item in found))

    def test_frontend_rejects_timestamp_payload_mismatch(self) -> None:
        frontend = StreamingAudioFrontend(AudioConfig())
        with self.assertRaises(ValueError):
            frontend.push(
                AudioChunk(
                    tenant_id="tenant",
                    session_id="session",
                    client_epoch=1,
                    seq_no=0,
                    start_ms=0,
                    end_ms=100,
                    pcm=b"\x00\x00" * 100,
                )
            )


if __name__ == "__main__":
    unittest.main()

