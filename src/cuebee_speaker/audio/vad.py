"""A dependency-free energy Voice Activity Detection (VAD) baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VADResult:
    frame_is_speech: np.ndarray
    speech_ratio: float
    root_mean_square_dbfs: np.ndarray


class EnergyVAD:
    """Classify fixed-size frames with an energy threshold and short hangover.

    This is a deliberately inspectable baseline. A production Web Real-Time
    Communication (WebRTC) or neural VAD can be injected without changing the
    streaming frontend contract.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        threshold_dbfs: float = -48.0,
        hangover_frames: int = 2,
    ) -> None:
        if sample_rate <= 0 or frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive")
        self.sample_rate = sample_rate
        self.frame_samples = max(1, sample_rate * frame_ms // 1_000)
        self.threshold_dbfs = threshold_dbfs
        self.hangover_frames = max(0, hangover_frames)

    def analyze(self, samples: np.ndarray) -> VADResult:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return VADResult(
                frame_is_speech=np.zeros((0,), dtype=bool),
                speech_ratio=0.0,
                root_mean_square_dbfs=np.zeros((0,), dtype=np.float32),
            )

        frame_count = int(np.ceil(audio.size / self.frame_samples))
        padded = np.pad(audio, (0, frame_count * self.frame_samples - audio.size))
        frames = padded.reshape(frame_count, self.frame_samples)
        root_mean_square = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        dbfs = 20.0 * np.log10(root_mean_square + 1e-12)
        speech = dbfs >= self.threshold_dbfs

        if self.hangover_frames and np.any(speech):
            kernel = np.ones((self.hangover_frames * 2 + 1,), dtype=np.int16)
            speech = np.convolve(speech.astype(np.int16), kernel, mode="same") > 0

        return VADResult(
            frame_is_speech=speech,
            speech_ratio=float(np.mean(speech)),
            root_mean_square_dbfs=dbfs.astype(np.float32),
        )

