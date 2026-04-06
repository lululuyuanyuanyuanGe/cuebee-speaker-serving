"""Session epoch and sequence-number admission for streaming audio events."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Tuple

from .domain import AudioChunk, GatewayDecision, GatewayStatus


@dataclass
class _Cursor:
    epoch: int
    last_seq: int


class EventGateway:
    """Deduplicates retransmits while allowing an explicit newer client epoch.

    A sequence gap is accepted and surfaced to the caller. This keeps the data plane
    available while allowing metrics or a control channel to request retransmission.
    """

    def __init__(self) -> None:
        self._cursors: Dict[Tuple[str, str], _Cursor] = {}
        self._lock = threading.Lock()

    def accept(self, chunk: AudioChunk) -> GatewayDecision:
        key = (chunk.tenant_id, chunk.session_id)
        with self._lock:
            cursor = self._cursors.get(key)
            if cursor is None or chunk.client_epoch > cursor.epoch:
                missing = (0, chunk.seq_no - 1) if chunk.seq_no > 0 else None
                status = GatewayStatus.GAP_ACCEPTED if missing else GatewayStatus.ACCEPTED
                self._cursors[key] = _Cursor(chunk.client_epoch, chunk.seq_no)
                return GatewayDecision(status, chunk.seq_no + 1, missing)

            if chunk.client_epoch < cursor.epoch:
                return GatewayDecision(GatewayStatus.STALE_EPOCH, cursor.last_seq + 1)

            if chunk.seq_no <= cursor.last_seq:
                return GatewayDecision(GatewayStatus.DUPLICATE, cursor.last_seq + 1)

            missing = None
            status = GatewayStatus.ACCEPTED
            if chunk.seq_no > cursor.last_seq + 1:
                missing = (cursor.last_seq + 1, chunk.seq_no - 1)
                status = GatewayStatus.GAP_ACCEPTED

            cursor.last_seq = chunk.seq_no
            return GatewayDecision(status, chunk.seq_no + 1, missing)

    def forget(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            self._cursors.pop((tenant_id, session_id), None)

