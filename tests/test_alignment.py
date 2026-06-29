import unittest

from cuebee_speaker.alignment import SpeakerTimeline, TimelineAligner
from cuebee_speaker.domain import SpeakerSegment, STTSegment, STTWord


def speaker(speaker_id: str, start_ms: int, end_ms: int, revision: int = 1):
    return SpeakerSegment(
        tenant_id="tenant",
        session_id="session",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_id=speaker_id,
        confidence=0.9,
        revision=revision,
    )


class AlignmentTests(unittest.TestCase):
    def test_splits_word_timestamps_at_speaker_change(self) -> None:
        event = STTSegment(
            tenant_id="tenant",
            session_id="session",
            segment_id="segment",
            revision=4,
            text="hello there yes indeed",
            start_ms=0,
            end_ms=2_000,
            is_final=True,
            words=(
                STTWord("hello", 0, 400),
                STTWord("there", 400, 900),
                STTWord("yes", 1_100, 1_400),
                STTWord("indeed", 1_400, 1_900),
            ),
        )
        spans = TimelineAligner().align(
            event, [speaker("spk_001", 0, 1_000), speaker("spk_002", 1_000, 2_000)]
        )
        self.assertEqual([item.speaker_id for item in spans], ["spk_001", "spk_002"])
        self.assertEqual([item.text for item in spans], ["hello there", "yes indeed"])
        self.assertTrue(all(item.revision == 4 for item in spans))

    def test_unknown_when_no_timeline_overlap(self) -> None:
        event = STTSegment(
            "tenant", "session", "segment", 1, "outside", 3_000, 4_000, True
        )
        span = TimelineAligner().align(event, [speaker("spk_001", 0, 1_000)])[0]
        self.assertEqual(span.speaker_id, "spk_unknown")

    def test_timeline_replaces_same_window_with_newer_revision(self) -> None:
        timeline = SpeakerTimeline()
        timeline.add([speaker("spk_001", 0, 1_000, revision=1)])
        timeline.add([speaker("spk_002", 0, 1_000, revision=2)])
        found = timeline.query("tenant", "session", 0, 1_000)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].speaker_id, "spk_002")


if __name__ == "__main__":
    unittest.main()

