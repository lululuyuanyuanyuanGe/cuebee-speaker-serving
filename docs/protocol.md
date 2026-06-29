# Streaming protocol

The reference gateway uses Newline-Delimited JSON (NDJSON) over a Transmission Control
Protocol (TCP) stream. It is deliberately dependency-free so the serving logic can be
tested without a web framework. Production ingress may translate WebSocket or Remote
Procedure Call (RPC) traffic into the same domain events.

## Audio event

```json
{
  "type": "audio",
  "tenant_id": "tenant-7",
  "session_id": "session-42",
  "client_epoch": 3,
  "seq_no": 81,
  "start_ms": 8100,
  "end_ms": 8200,
  "sample_rate": 16000,
  "channels": 1,
  "is_final": false,
  "pcm_base64": "BASE64_SIGNED_PCM16LE"
}
```

Successful response:

```json
{
  "ok": true,
  "gateway": "accepted",
  "expected_next_seq": 82,
  "missing_range": null,
  "speaker_segments": []
}
```

Speaker segments appear only when enough buffered speech forms a window. A
`gap_accepted` response includes the inclusive missing sequence range. `duplicate` and
`stale_epoch` events do not enter the frontend.

## STT event

```json
{
  "type": "stt",
  "tenant_id": "tenant-7",
  "session_id": "session-42",
  "segment_id": "stt-19",
  "revision": 4,
  "text": "Alice reviews it Bob agrees",
  "start_ms": 7000,
  "end_ms": 9500,
  "is_final": true,
  "words": [
    {"text": "Alice", "start_ms": 7000, "end_ms": 7300},
    {"text": "reviews", "start_ms": 7300, "end_ms": 7900},
    {"text": "it", "start_ms": 7900, "end_ms": 8100},
    {"text": "Bob", "start_ms": 8500, "end_ms": 8800},
    {"text": "agrees", "start_ms": 8800, "end_ms": 9400}
  ]
}
```

When word timestamps are absent, the entire STT segment receives the speaker with maximum
overlap. When no speaker window overlaps, the identifier is `spk_unknown`.

## Metadata and control events

- `{"type":"health"}` returns the active backend name.
- `{"type":"metrics"}` returns counters, gauges, and percentile summaries.
- `{"type":"profiles","tenant_id":"...","session_id":"..."}` returns profile
  metadata but never returns centroid values.
- `{"type":"rename",...}` changes the display name only.
- `{"type":"flush",...}` flushes a sufficiently long final buffered window.
- `{"type":"autoscale",...}` returns a pure scaling recommendation.

## Trust boundary

The reference TCP server does not authenticate `tenant_id`. A production edge must derive
the tenant from an authenticated claim, reject a mismatching message field, terminate
Transport Layer Security (TLS), enforce message and rate limits, and avoid logging full
audio or transcript bodies.

