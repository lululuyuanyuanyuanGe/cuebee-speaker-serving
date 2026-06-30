"""Validated configuration for the speaker-serving data and control planes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_ms: int = 30
    vad_threshold_dbfs: float = -48.0
    min_speech_ratio: float = 0.30
    window_ms: int = 1_500
    hop_ms: int = 750
    fbank_bins: int = 80
    fbank_frame_ms: int = 25
    fbank_hop_ms: int = 10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= self.min_speech_ratio <= 1.0:
            raise ValueError("min_speech_ratio must be in [0, 1]")
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("window_ms and hop_ms must be positive")
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms cannot exceed window_ms")
        if self.fbank_bins <= 0:
            raise ValueError("fbank_bins must be positive")


@dataclass(frozen=True)
class BatcherConfig:
    max_batch_size: int = 8
    max_wait_ms: int = 80
    per_session_batch_limit: int = 2
    bucket_boundaries_frames: Tuple[int, ...] = (200, 400)
    default_deadline_ms: int = 250

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_wait_ms <= 0 or self.default_deadline_ms <= 0:
            raise ValueError("wait and deadline values must be positive")
        if self.per_session_batch_limit <= 0:
            raise ValueError("per_session_batch_limit must be positive")
        if tuple(sorted(self.bucket_boundaries_frames)) != self.bucket_boundaries_frames:
            raise ValueError("bucket boundaries must be sorted")


@dataclass(frozen=True)
class AssignmentConfig:
    cosine_threshold: float = 0.72
    min_quality: float = 0.20
    quality_ema_alpha: float = 0.10
    max_centroid_weight: float = 20.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if not 0.0 < self.quality_ema_alpha <= 1.0:
            raise ValueError("quality_ema_alpha must be in (0, 1]")


@dataclass(frozen=True)
class AutoscalerConfig:
    min_replicas: int = 2
    max_replicas: int = 32
    target_utilization: float = 0.70
    backlog_high_seconds: float = 8.0
    backlog_low_seconds: float = 1.0
    queue_high_ms: float = 120.0
    cooldown_seconds: float = 30.0
    scale_down_windows: int = 3

    def __post_init__(self) -> None:
        if self.min_replicas <= 0 or self.max_replicas < self.min_replicas:
            raise ValueError("invalid replica bounds")
        if not 0.0 < self.target_utilization < 1.0:
            raise ValueError("target_utilization must be in (0, 1)")
        if self.scale_down_windows <= 0:
            raise ValueError("scale_down_windows must be positive")


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    batcher: BatcherConfig = field(default_factory=BatcherConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    autoscaler: AutoscalerConfig = field(default_factory=AutoscalerConfig)
    model_path: Optional[str] = None
    native_worker_path: Optional[str] = None
    native_worker_backend: str = "auto"
    native_worker_timeout_seconds: float = 5.0
    native_worker_intra_op_threads: int = 0
    native_worker_restart_attempts: int = 1
    state_db_path: str = "runtime/speaker-state.sqlite3"

    def __post_init__(self) -> None:
        if self.native_worker_backend not in {"auto", "deterministic", "onnx"}:
            raise ValueError("native_worker_backend must be auto, deterministic, or onnx")
        if self.native_worker_timeout_seconds <= 0:
            raise ValueError("native_worker_timeout_seconds must be positive")
        if self.native_worker_intra_op_threads < 0:
            raise ValueError("native_worker_intra_op_threads cannot be negative")
        if self.native_worker_restart_attempts < 0:
            raise ValueError("native_worker_restart_attempts cannot be negative")
