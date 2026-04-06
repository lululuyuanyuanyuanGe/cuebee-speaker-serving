"""Domain events shared across gateway, inference, state, and alignment layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class GatewayStatus(str, Enum):
    ACCEPTED = "accepted"
    GAP_ACCEPTED = "gap_accepted"
    DUPLICATE = "duplicate"
    STALE_EPOCH = "stale_epoch"


@dataclass(frozen=True)
class GatewayDecision:
    status: GatewayStatus
    expected_next_seq: int
    missing_range: Optional[Tuple[int, int]] = None

    @property
    def accepted(self) -> bool:
        return self.status in (GatewayStatus.ACCEPTED, GatewayStatus.GAP_ACCEPTED)


@dataclass(frozen=True)
class AudioChunk:
    tenant_id: str
    session_id: str
    client_epoch: int
    seq_no: int
    start_ms: int
    end_ms: int
    pcm: bytes
    sample_rate: int = 16_000
    channels: int = 1
    is_final: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.session_id:
            raise ValueError("tenant_id and session_id are required")
        if self.client_epoch < 0 or self.seq_no < 0:
            raise ValueError("client_epoch and seq_no must be non-negative")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.sample_rate <= 0 or self.channels != 1:
            raise ValueError("only positive-rate mono audio is supported")
        if len(self.pcm) % 2:
            raise ValueError("PCM signed 16-bit little-endian payload must have even length")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class FeatureChunk:
    tenant_id: str
    session_id: str
    chunk_id: str
    start_ms: int
    end_ms: int
    features: np.ndarray
    speech_ratio: float

    @property
    def audio_seconds(self) -> float:
        return max(0.0, (self.end_ms - self.start_ms) / 1_000.0)


@dataclass
class EmbeddingRequest:
    chunk: FeatureChunk
    deadline_at: float
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class EmbeddingResult:
    chunk: FeatureChunk
    embedding: np.ndarray
    quality: float
    worker_id: str
    batch_size: int
    queue_delay_ms: float
    inference_ms: float


@dataclass
class SpeakerProfile:
    tenant_id: str
    session_id: str
    speaker_id: str
    centroid: np.ndarray
    sample_count: int
    quality_ema: float
    last_active_ms: int
    display_name: Optional[str] = None


@dataclass(frozen=True)
class SpeakerSegment:
    tenant_id: str
    session_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    confidence: float
    revision: int = 1


@dataclass(frozen=True)
class STTWord:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class STTSegment:
    tenant_id: str
    session_id: str
    segment_id: str
    revision: int
    text: str
    start_ms: int
    end_ms: int
    is_final: bool
    words: Sequence[STTWord] = ()


@dataclass(frozen=True)
class AttributedTranscriptSpan:
    tenant_id: str
    session_id: str
    segment_id: str
    revision: int
    speaker_id: str
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class IngestResult:
    gateway: GatewayDecision
    speaker_segments: List[SpeakerSegment]

