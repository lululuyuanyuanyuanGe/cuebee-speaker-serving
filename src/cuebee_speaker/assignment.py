"""Online nearest-centroid speaker assignment with stable session identifiers."""

from __future__ import annotations

import asyncio
import threading
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import AssignmentConfig
from .domain import EmbeddingResult, SpeakerProfile, SpeakerSegment
from .state import SpeakerStore


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding must have a finite non-zero norm")
    return value / norm


class OnlineSpeakerAssigner:
    def __init__(self, store: SpeakerStore, config: AssignmentConfig) -> None:
        self._store = store
        self._config = config
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._locks_guard = threading.Lock()

    async def assign(self, result: EmbeddingResult) -> SpeakerSegment:
        chunk = result.chunk
        if result.quality < self._config.min_quality:
            return SpeakerSegment(
                tenant_id=chunk.tenant_id,
                session_id=chunk.session_id,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                speaker_id="spk_unknown",
                confidence=result.quality,
            )

        embedding = _normalize(result.embedding)
        lock = self._get_lock(chunk.tenant_id, chunk.session_id)
        async with lock:
            profiles = await asyncio.to_thread(
                self._store.list_profiles, chunk.tenant_id, chunk.session_id
            )
            selected, score = self._nearest(profiles, embedding)
            if selected is None or score < self._config.cosine_threshold:
                selected = await asyncio.to_thread(
                    self._store.create_profile,
                    chunk.tenant_id,
                    chunk.session_id,
                    embedding,
                    result.quality,
                    chunk.end_ms,
                )
                confidence = result.quality
            else:
                selected = self._updated_profile(selected, embedding, result.quality, chunk.end_ms)
                await asyncio.to_thread(self._store.update_profile, selected)
                confidence = score
        return SpeakerSegment(
            tenant_id=chunk.tenant_id,
            session_id=chunk.session_id,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            speaker_id=selected.speaker_id,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    async def rename(
        self, tenant_id: str, session_id: str, speaker_id: str, display_name: str
    ) -> bool:
        lock = self._get_lock(tenant_id, session_id)
        async with lock:
            return await asyncio.to_thread(
                self._store.rename_speaker,
                tenant_id,
                session_id,
                speaker_id,
                display_name,
            )

    async def profiles(self, tenant_id: str, session_id: str) -> Tuple[SpeakerProfile, ...]:
        profiles = await asyncio.to_thread(self._store.list_profiles, tenant_id, session_id)
        return tuple(profiles)

    async def close(self) -> None:
        await asyncio.to_thread(self._store.close)

    def _get_lock(self, tenant_id: str, session_id: str) -> asyncio.Lock:
        key = (tenant_id, session_id)
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @staticmethod
    def _nearest(
        profiles: Sequence[SpeakerProfile], embedding: np.ndarray
    ) -> Tuple[Optional[SpeakerProfile], float]:
        if not profiles:
            return None, -1.0
        scores = [float(np.dot(_normalize(profile.centroid), embedding)) for profile in profiles]
        index = int(np.argmax(scores))
        return profiles[index], scores[index]

    def _updated_profile(
        self,
        profile: SpeakerProfile,
        embedding: np.ndarray,
        quality: float,
        last_active_ms: int,
    ) -> SpeakerProfile:
        existing_weight = min(
            float(profile.sample_count), self._config.max_centroid_weight
        ) * max(profile.quality_ema, 0.05)
        new_weight = max(quality, 0.05)
        centroid = _normalize(profile.centroid * existing_weight + embedding * new_weight)
        alpha = self._config.quality_ema_alpha
        return SpeakerProfile(
            tenant_id=profile.tenant_id,
            session_id=profile.session_id,
            speaker_id=profile.speaker_id,
            centroid=centroid,
            sample_count=profile.sample_count + 1,
            quality_ema=(1.0 - alpha) * profile.quality_ema + alpha * quality,
            last_active_ms=last_active_ms,
            display_name=profile.display_name,
        )
