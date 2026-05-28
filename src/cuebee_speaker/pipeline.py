"""End-to-end orchestration for audio ingestion, assignment, and STT alignment."""

from __future__ import annotations

import asyncio
import threading
from typing import Dict, List, Optional, Tuple

from .alignment import SpeakerTimeline, TimelineAligner
from .assignment import OnlineSpeakerAssigner
from .audio import StreamingAudioFrontend
from .domain import (
    AttributedTranscriptSpan,
    AudioChunk,
    FeatureChunk,
    IngestResult,
    SpeakerSegment,
    STTSegment,
)
from .gateway import EventGateway
from .inference import CrossSessionMicroBatcher


class DiarizationPipeline:
    def __init__(
        self,
        gateway: EventGateway,
        frontend: StreamingAudioFrontend,
        batcher: CrossSessionMicroBatcher,
        assigner: OnlineSpeakerAssigner,
        timeline: Optional[SpeakerTimeline] = None,
        aligner: Optional[TimelineAligner] = None,
    ) -> None:
        self.gateway = gateway
        self.frontend = frontend
        self.batcher = batcher
        self.assigner = assigner
        self.timeline = timeline or SpeakerTimeline()
        self.aligner = aligner or TimelineAligner()
        self._session_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._session_locks_guard = threading.Lock()

    async def ingest_audio(self, chunk: AudioChunk) -> IngestResult:
        lock = self._get_session_lock(chunk.tenant_id, chunk.session_id)
        async with lock:
            decision = self.gateway.accept(chunk)
            if not decision.accepted:
                return IngestResult(gateway=decision, speaker_segments=[])
            features = self.frontend.push(chunk)
            if chunk.is_final:
                features.extend(self.frontend.flush(chunk.tenant_id, chunk.session_id))
            segments = await self._process_features(features)
            self.timeline.add(segments)
            return IngestResult(gateway=decision, speaker_segments=segments)

    async def flush_session(self, tenant_id: str, session_id: str) -> List[SpeakerSegment]:
        lock = self._get_session_lock(tenant_id, session_id)
        async with lock:
            features = self.frontend.flush(tenant_id, session_id)
            segments = await self._process_features(features)
            self.timeline.add(segments)
            return segments

    def align_stt(self, event: STTSegment) -> List[AttributedTranscriptSpan]:
        speakers = self.timeline.query(
            event.tenant_id, event.session_id, event.start_ms, event.end_ms
        )
        return self.aligner.align(event, speakers)

    async def rename_speaker(
        self, tenant_id: str, session_id: str, speaker_id: str, display_name: str
    ) -> bool:
        return await self.assigner.rename(tenant_id, session_id, speaker_id, display_name)

    async def close(self) -> None:
        await self.batcher.close()
        await self.assigner.close()

    async def _process_features(
        self, features: List[FeatureChunk]
    ) -> List[SpeakerSegment]:
        if not features:
            return []
        results = await asyncio.gather(*(self.batcher.submit(item) for item in features))
        ordered = sorted(results, key=lambda item: (item.chunk.start_ms, item.chunk.end_ms))
        segments = []
        for result in ordered:
            segments.append(await self.assigner.assign(result))
        return segments

    def _get_session_lock(self, tenant_id: str, session_id: str) -> asyncio.Lock:
        key = (tenant_id, session_id)
        with self._session_locks_guard:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[key] = lock
            return lock
