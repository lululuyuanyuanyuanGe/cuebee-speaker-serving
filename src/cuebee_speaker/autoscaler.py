"""Audio-workload autoscaling recommendation with hysteresis and cooldown."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

from .config import AutoscalerConfig


@dataclass(frozen=True)
class ScalingSample:
    audio_arrival_seconds_per_second: float
    real_time_factor: float
    backlog_seconds: float
    queue_p95_ms: float
    worker_utilization: float
    current_replicas: int
    observed_at: float = 0.0


@dataclass(frozen=True)
class ScalingDecision:
    current_replicas: int
    desired_replicas: int
    raw_capacity_replicas: int
    action: str
    reasons: Tuple[str, ...]


class AudioBacklogAutoscaler:
    """Recommend replicas from audio rate, Real-Time Factor (RTF), and backlog."""

    def __init__(self, config: AutoscalerConfig) -> None:
        self._config = config
        self._last_action_at = float("-inf")
        self._low_windows = 0

    def recommend(self, sample: ScalingSample) -> ScalingDecision:
        self._validate(sample)
        now = sample.observed_at or time.monotonic()
        raw = math.ceil(
            sample.audio_arrival_seconds_per_second
            * sample.real_time_factor
            / self._config.target_utilization
        )
        raw = self._clamp(raw)
        desired = raw
        reasons = ["audio_capacity"]

        if sample.backlog_seconds >= self._config.backlog_high_seconds:
            correction = max(
                1, math.ceil(sample.backlog_seconds / self._config.backlog_high_seconds)
            )
            desired = max(desired, sample.current_replicas + correction)
            reasons.append("backlog_high")
        if sample.queue_p95_ms >= self._config.queue_high_ms:
            desired = max(desired, sample.current_replicas + 1)
            reasons.append("queue_p95_high")
        desired = self._clamp(desired)

        cooldown_elapsed = now - self._last_action_at >= self._config.cooldown_seconds
        if desired > sample.current_replicas:
            self._low_windows = 0
            if cooldown_elapsed:
                self._last_action_at = now
                return ScalingDecision(
                    sample.current_replicas, desired, raw, "scale_up", tuple(reasons)
                )
            return ScalingDecision(
                sample.current_replicas,
                sample.current_replicas,
                raw,
                "hold",
                tuple(reasons + ["cooldown"]),
            )

        low_load = (
            desired < sample.current_replicas
            and sample.backlog_seconds <= self._config.backlog_low_seconds
            and sample.queue_p95_ms < self._config.queue_high_ms * 0.5
            and sample.worker_utilization < self._config.target_utilization * 0.65
        )
        if low_load:
            self._low_windows += 1
        else:
            self._low_windows = 0

        if (
            low_load
            and self._low_windows >= self._config.scale_down_windows
            and cooldown_elapsed
        ):
            target = max(desired, sample.current_replicas - 1)
            target = self._clamp(target)
            self._last_action_at = now
            self._low_windows = 0
            return ScalingDecision(
                sample.current_replicas,
                target,
                raw,
                "scale_down",
                ("sustained_low_load",),
            )

        hold_reasons = reasons
        if low_load:
            hold_reasons.append("scale_down_hysteresis")
        return ScalingDecision(
            sample.current_replicas,
            sample.current_replicas,
            raw,
            "hold",
            tuple(hold_reasons),
        )

    def _clamp(self, replicas: int) -> int:
        return min(self._config.max_replicas, max(self._config.min_replicas, replicas))

    @staticmethod
    def _validate(sample: ScalingSample) -> None:
        if sample.audio_arrival_seconds_per_second < 0:
            raise ValueError("audio arrival rate cannot be negative")
        if sample.real_time_factor <= 0:
            raise ValueError("real_time_factor must be positive")
        if sample.backlog_seconds < 0 or sample.queue_p95_ms < 0:
            raise ValueError("backlog and queue delay cannot be negative")
        if not 0.0 <= sample.worker_utilization <= 1.0:
            raise ValueError("worker_utilization must be in [0, 1]")
        if sample.current_replicas <= 0:
            raise ValueError("current_replicas must be positive")

