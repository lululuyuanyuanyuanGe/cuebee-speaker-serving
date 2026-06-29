"""NumPy log-Mel filter-bank extraction for the ERes2Net input contract."""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import numpy as np


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2_595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2_595.0) - 1.0)


@functools.lru_cache(maxsize=16)
def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(np.asarray(low_hz)),
        _hz_to_mel(np.asarray(high_hz)),
        n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index, frequency_bin] = (frequency_bin - left) / (center - left)
        for frequency_bin in range(center, min(right + 1, filters.shape[1])):
            filters[index, frequency_bin] = (right - frequency_bin) / (right - center)
    return filters


class LogMelFbank:
    """Extract `[time, mel]` float32 features from 16 kilohertz mono audio."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        frame_ms: int = 25,
        hop_ms: int = 10,
        n_fft: int = 512,
        preemphasis: float = 0.97,
        normalize_mean: bool = True,
    ) -> None:
        if sample_rate <= 0 or n_mels <= 0 or n_fft <= 0:
            raise ValueError("filter-bank dimensions must be positive")
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_samples = sample_rate * frame_ms // 1_000
        self.hop_samples = sample_rate * hop_ms // 1_000
        self.n_fft = n_fft
        self.preemphasis = preemphasis
        self.normalize_mean = normalize_mean
        if self.n_fft < self.frame_samples:
            raise ValueError("n_fft cannot be smaller than the analysis frame")
        self._window = np.hanning(self.frame_samples).astype(np.float32)
        self._filters = _mel_filterbank(
            sample_rate, n_fft, n_mels, 20.0, sample_rate / 2.0
        )

    def extract(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("cannot extract features from empty audio")
        if audio.size < self.frame_samples:
            audio = np.pad(audio, (0, self.frame_samples - audio.size))

        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.preemphasis * audio[:-1]

        frame_count = 1 + int(np.ceil((emphasized.size - self.frame_samples) / self.hop_samples))
        total = (frame_count - 1) * self.hop_samples + self.frame_samples
        emphasized = np.pad(emphasized, (0, total - emphasized.size))
        indices = (
            np.arange(self.frame_samples, dtype=np.int64)[None, :]
            + self.hop_samples * np.arange(frame_count, dtype=np.int64)[:, None]
        )
        frames = emphasized[indices] * self._window[None, :]
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power = (np.abs(spectrum) ** 2 / self.n_fft).astype(np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel_energy = power @ self._filters.T
        if not np.all(np.isfinite(mel_energy)):
            raise FloatingPointError("non-finite log-Mel energy produced by the frontend")
        mel_energy = np.maximum(mel_energy, 1e-10)
        features = np.log(mel_energy).astype(np.float32)
        if self.normalize_mean:
            features -= np.mean(features, axis=0, keepdims=True)
        return np.ascontiguousarray(features, dtype=np.float32)

    @property
    def output_shape(self) -> Tuple[Optional[int], int]:
        return (None, self.n_mels)
