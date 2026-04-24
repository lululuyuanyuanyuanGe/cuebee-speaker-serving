"""Pulse-Code Modulation (PCM) conversion helpers."""

from __future__ import annotations

import numpy as np


def pcm16le_to_float32(payload: bytes) -> np.ndarray:
    """Decode mono signed 16-bit little-endian PCM into the [-1, 1) range."""

    if len(payload) % 2:
        raise ValueError("PCM signed 16-bit payload must contain complete samples")
    if not payload:
        return np.empty((0,), dtype=np.float32)
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32_768.0


def float32_to_pcm16le(samples: np.ndarray) -> bytes:
    """Encode floating-point audio as mono signed 16-bit little-endian PCM."""

    values = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(values, -1.0, 1.0 - (1.0 / 32_768.0))
    return np.rint(clipped * 32_768.0).astype("<i2").tobytes()

