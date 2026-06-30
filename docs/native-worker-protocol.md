# Native worker protocol

The Python gateway and C++ inference worker communicate over the worker's standard input
and standard output. Standard error is reserved for diagnostics. The transport is local,
ordered, length-prefixed, and intentionally has no third-party serialization dependency.
All integers and 32-bit floating-point values are little-endian.

## Request

The fixed 32-byte header is:

| Offset | Type | Field |
|---:|---|---|
| 0 | 4 bytes | magic `CBSP` |
| 4 | unsigned 16-bit | protocol version, currently 1 |
| 6 | unsigned 16-bit | operation: infer 1, ping 2, shutdown 3 |
| 8 | unsigned 64-bit | monotonically increasing request identifier |
| 16 | unsigned 32-bit | batch size |
| 20 | unsigned 32-bit | log-Mel bins, currently 80 |
| 24 | unsigned 32-bit | payload byte count |
| 28 | unsigned 32-bit | reserved, must be zero |

An inference payload repeats `frame_count: uint32` followed by
`frame_count * 80` contiguous float32 values for each item. Control messages have an empty
payload. The worker rejects empty inference batches, non-80-bin features, oversized batches,
oversized frame counts, truncated values, and trailing bytes before calling the model.

## Response

The fixed 32-byte header is:

| Offset | Type | Field |
|---:|---|---|
| 0 | 4 bytes | magic `CBSR` |
| 4 | unsigned 16-bit | protocol version |
| 6 | unsigned 16-bit | status: success 0, invalid 1, backend 2, unsupported 3 |
| 8 | unsigned 64-bit | matching request identifier |
| 16 | unsigned 32-bit | result batch size |
| 20 | unsigned 32-bit | embedding dimension, 192 for inference |
| 24 | unsigned 32-bit | payload byte count |
| 28 | unsigned 32-bit | leading error-message byte count |

On success, each result is one quality float followed by 192 embedding floats. Embeddings
are finite and length-normalized in the C++ worker. On failure, the payload begins with a
UTF-8 (Unicode Transformation Format 8-bit) error string and has no results.

Only one request is in flight per worker process. Cross-session concurrency is consolidated
by the upstream micro-batcher; horizontal concurrency comes from multiple stateless worker
processes. A response timeout or closed pipe causes the Python adapter to terminate the old
process, start a new one, perform the next request, and preserve authoritative speaker state
in the store.
