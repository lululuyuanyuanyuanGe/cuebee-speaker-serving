"""Timeline storage and Speech-to-Text (STT) speaker attribution."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .domain import AttributedTranscriptSpan, SpeakerSegment, STTSegment, STTWord


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


class SpeakerTimeline:
    def __init__(self, retention_ms: int = 2 * 60 * 60 * 1_000) -> None:
        self._retention_ms = retention_ms
        self._segments: DefaultDict[Tuple[str, str], List[SpeakerSegment]] = defaultdict(list)
        self._lock = threading.RLock()

    def add(self, segments: Iterable[SpeakerSegment]) -> None:
        grouped: DefaultDict[Tuple[str, str], List[SpeakerSegment]] = defaultdict(list)
        for segment in segments:
            grouped[(segment.tenant_id, segment.session_id)].append(segment)
        with self._lock:
            for key, new_segments in grouped.items():
                current: Dict[Tuple[int, int], SpeakerSegment] = {
                    (item.start_ms, item.end_ms): item for item in self._segments[key]
                }
                for segment in new_segments:
                    slot = (segment.start_ms, segment.end_ms)
                    prior = current.get(slot)
                    if prior is None or segment.revision >= prior.revision:
                        current[slot] = segment
                ordered = sorted(current.values(), key=lambda item: (item.start_ms, item.end_ms))
                if ordered:
                    cutoff = ordered[-1].end_ms - self._retention_ms
                    ordered = [item for item in ordered if item.end_ms >= cutoff]
                self._segments[key] = ordered

    def query(
        self, tenant_id: str, session_id: str, start_ms: int, end_ms: int
    ) -> Tuple[SpeakerSegment, ...]:
        with self._lock:
            return tuple(
                item
                for item in self._segments.get((tenant_id, session_id), ())
                if _overlap_ms(start_ms, end_ms, item.start_ms, item.end_ms) > 0
            )

    def forget(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            self._segments.pop((tenant_id, session_id), None)


class TimelineAligner:
    """Assign words, or a whole STT segment, by maximum timestamp overlap."""

    def align(
        self, stt: STTSegment, speakers: Sequence[SpeakerSegment]
    ) -> List[AttributedTranscriptSpan]:
        compatible = [
            segment
            for segment in speakers
            if segment.tenant_id == stt.tenant_id and segment.session_id == stt.session_id
        ]
        if not stt.words:
            speaker_id = self._best_speaker(stt.start_ms, stt.end_ms, compatible)
            return [
                AttributedTranscriptSpan(
                    tenant_id=stt.tenant_id,
                    session_id=stt.session_id,
                    segment_id=stt.segment_id,
                    revision=stt.revision,
                    speaker_id=speaker_id,
                    text=stt.text,
                    start_ms=stt.start_ms,
                    end_ms=stt.end_ms,
                )
            ]

        assignments = [
            (word, self._best_speaker(word.start_ms, word.end_ms, compatible))
            for word in stt.words
        ]
        groups: List[Tuple[str, List[STTWord]]] = []
        for word, speaker_id in assignments:
            if groups and groups[-1][0] == speaker_id:
                groups[-1][1].append(word)
            else:
                groups.append((speaker_id, [word]))
        return [
            AttributedTranscriptSpan(
                tenant_id=stt.tenant_id,
                session_id=stt.session_id,
                segment_id=stt.segment_id,
                revision=stt.revision,
                speaker_id=speaker_id,
                text=self._join_words(words),
                start_ms=words[0].start_ms,
                end_ms=words[-1].end_ms,
            )
            for speaker_id, words in groups
        ]

    @staticmethod
    def _best_speaker(
        start_ms: int, end_ms: int, speakers: Sequence[SpeakerSegment]
    ) -> str:
        if not speakers:
            return "spk_unknown"
        ranked = sorted(
            speakers,
            key=lambda item: (
                _overlap_ms(start_ms, end_ms, item.start_ms, item.end_ms),
                item.confidence,
                -item.start_ms,
            ),
            reverse=True,
        )
        if _overlap_ms(start_ms, end_ms, ranked[0].start_ms, ranked[0].end_ms) <= 0:
            return "spk_unknown"
        return ranked[0].speaker_id

    @staticmethod
    def _join_words(words: Sequence[STTWord]) -> str:
        text = ""
        no_leading_space = set(",.!?;:，。！？；：)]}")
        for word in words:
            value = word.text
            if not text or value[:1] in no_leading_space or text[-1:].isspace():
                text += value
            elif TimelineAligner._is_cjk(text[-1]) or TimelineAligner._is_cjk(value[0]):
                text += value
            else:
                text += " " + value
        return text

    @staticmethod
    def _is_cjk(character: str) -> bool:
        return "\u3400" <= character <= "\u9fff"

