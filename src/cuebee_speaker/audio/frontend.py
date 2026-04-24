"""Per-session streaming window assembly and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..config import AudioConfig
from ..domain import AudioChunk, FeatureChunk
from .fbank import LogMelFbank
from .pcm import pcm16le_to_float32
from .vad import EnergyVAD


@dataclass
class _StreamState:
    epoch: int
    audio: np.ndarray
    base_ms: int
    expected_end_ms: int
    emitted: int = 0


class StreamingAudioFrontend:
    """Turn timestamped PCM packets into overlapping speech feature windows."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._window_samples = config.sample_rate * config.window_ms // 1_000
        self._hop_samples = config.sample_rate * config.hop_ms // 1_000
        self._states: Dict[Tuple[str, str], _StreamState] = {}
        self._vad = EnergyVAD(
            sample_rate=config.sample_rate,
            frame_ms=config.frame_ms,
            threshold_dbfs=config.vad_threshold_dbfs,
        )
        self._fbank = LogMelFbank(
            sample_rate=config.sample_rate,
            n_mels=config.fbank_bins,
            frame_ms=config.fbank_frame_ms,
            hop_ms=config.fbank_hop_ms,
        )

    def push(self, chunk: AudioChunk) -> List[FeatureChunk]:
        if chunk.sample_rate != self.config.sample_rate:
            raise ValueError(
                "sample-rate conversion is not implicit; send audio at "
                f"{self.config.sample_rate} hertz"
            )
        samples = pcm16le_to_float32(chunk.pcm)
        expected_samples = round(chunk.duration_ms * chunk.sample_rate / 1_000)
        tolerance = max(2, int(expected_samples * 0.02))
        if abs(samples.size - expected_samples) > tolerance:
            raise ValueError("PCM payload duration disagrees with the timestamp range")

        key = (chunk.tenant_id, chunk.session_id)
        state = self._states.get(key)
        if state is None or state.epoch != chunk.client_epoch:
            state = _StreamState(
                epoch=chunk.client_epoch,
                audio=np.empty((0,), dtype=np.float32),
                base_ms=chunk.start_ms,
                expected_end_ms=chunk.start_ms,
            )
            self._states[key] = state

        gap_ms = chunk.start_ms - state.expected_end_ms
        if gap_ms > self.config.frame_ms:
            state.audio = np.empty((0,), dtype=np.float32)
            state.base_ms = chunk.start_ms
        elif gap_ms < 0:
            overlap_samples = round(-gap_ms * self.config.sample_rate / 1_000)
            if overlap_samples >= samples.size:
                return []
            samples = samples[overlap_samples:]

        state.audio = np.concatenate((state.audio, samples))
        state.expected_end_ms = max(state.expected_end_ms, chunk.end_ms)
        return self._emit_complete_windows(chunk.tenant_id, chunk.session_id, state)

    def _emit_complete_windows(
        self, tenant_id: str, session_id: str, state: _StreamState
    ) -> List[FeatureChunk]:
        emitted: List[FeatureChunk] = []
        while state.audio.size >= self._window_samples:
            window = state.audio[: self._window_samples]
            start_ms = state.base_ms
            end_ms = start_ms + self.config.window_ms
            result = self._vad.analyze(window)
            if result.speech_ratio >= self.config.min_speech_ratio:
                emitted.append(
                    self._make_feature_chunk(
                        tenant_id,
                        session_id,
                        state,
                        start_ms,
                        end_ms,
                        window,
                        result.speech_ratio,
                    )
                )
            state.audio = state.audio[self._hop_samples :]
            state.base_ms += self.config.hop_ms
        return emitted

    def flush(self, tenant_id: str, session_id: str) -> List[FeatureChunk]:
        key = (tenant_id, session_id)
        state = self._states.pop(key, None)
        if state is None:
            return []
        minimum = max(self._fbank.frame_samples, self.config.sample_rate * 300 // 1_000)
        if state.audio.size < minimum:
            return []
        result = self._vad.analyze(state.audio)
        if result.speech_ratio < self.config.min_speech_ratio:
            return []
        duration_ms = round(state.audio.size * 1_000 / self.config.sample_rate)
        return [
            self._make_feature_chunk(
                tenant_id,
                session_id,
                state,
                state.base_ms,
                state.base_ms + duration_ms,
                state.audio,
                result.speech_ratio,
            )
        ]

    def _make_feature_chunk(
        self,
        tenant_id: str,
        session_id: str,
        state: _StreamState,
        start_ms: int,
        end_ms: int,
        samples: np.ndarray,
        speech_ratio: float,
    ) -> FeatureChunk:
        chunk_id = f"{tenant_id}:{session_id}:{state.epoch}:{state.emitted:08d}"
        state.emitted += 1
        return FeatureChunk(
            tenant_id=tenant_id,
            session_id=session_id,
            chunk_id=chunk_id,
            start_ms=start_ms,
            end_ms=end_ms,
            features=self._fbank.extract(samples),
            speech_ratio=speech_ratio,
        )

    def forget(self, tenant_id: str, session_id: str) -> None:
        self._states.pop((tenant_id, session_id), None)

