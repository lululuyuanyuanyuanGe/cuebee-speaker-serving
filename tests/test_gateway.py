import unittest

from cuebee_speaker.domain import AudioChunk, GatewayStatus
from cuebee_speaker.gateway import EventGateway


def chunk(tenant: str, session: str, epoch: int, sequence: int) -> AudioChunk:
    return AudioChunk(
        tenant_id=tenant,
        session_id=session,
        client_epoch=epoch,
        seq_no=sequence,
        start_ms=sequence * 100,
        end_ms=(sequence + 1) * 100,
        pcm=b"\x00\x00" * 1_600,
    )


class EventGatewayTests(unittest.TestCase):
    def test_accepts_in_order_and_deduplicates_retransmit(self) -> None:
        gateway = EventGateway()
        first = gateway.accept(chunk("tenant-a", "session", 1, 0))
        duplicate = gateway.accept(chunk("tenant-a", "session", 1, 0))
        second = gateway.accept(chunk("tenant-a", "session", 1, 1))

        self.assertEqual(first.status, GatewayStatus.ACCEPTED)
        self.assertEqual(duplicate.status, GatewayStatus.DUPLICATE)
        self.assertEqual(second.expected_next_seq, 2)

    def test_surfaces_sequence_gap_and_rejects_old_epoch(self) -> None:
        gateway = EventGateway()
        gateway.accept(chunk("tenant", "session", 3, 2))
        gap = gateway.accept(chunk("tenant", "session", 3, 5))
        stale = gateway.accept(chunk("tenant", "session", 2, 100))

        self.assertEqual(gap.status, GatewayStatus.GAP_ACCEPTED)
        self.assertEqual(gap.missing_range, (3, 4))
        self.assertEqual(stale.status, GatewayStatus.STALE_EPOCH)

    def test_tenant_is_part_of_cursor_identity(self) -> None:
        gateway = EventGateway()
        one = gateway.accept(chunk("tenant-a", "same-session", 1, 0))
        two = gateway.accept(chunk("tenant-b", "same-session", 1, 0))
        self.assertTrue(one.accepted)
        self.assertTrue(two.accepted)


if __name__ == "__main__":
    unittest.main()

