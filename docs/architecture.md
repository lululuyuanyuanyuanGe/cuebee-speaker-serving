# Architecture and invariants

## Scope boundary

The service owns the audio-to-speaker timeline. It receives timestamped Pulse-Code
Modulation (PCM) packets and third-party Speech-to-Text (STT) events. It does not recognize
text, train the speaker model, or implement CueBee's Large Language Model (LLM) runtime.

## Data-plane stages

1. The event gateway keys admission by `(tenant_id, session_id, client_epoch, seq_no)`.
   Retransmits are idempotent. A higher client epoch starts a new stream; an older epoch is
   rejected. Sequence gaps are accepted but surfaced.
2. The streaming frontend verifies timestamp duration against payload length, resets after
   a material gap, runs energy Voice Activity Detection (VAD), and emits 1.5-second windows
   with a 0.75-second hop by default.
3. The filter-bank frontend produces `[T,80]` float32 log-Mel features with per-bin mean
   normalization.
4. The micro-batcher isolates long requests in separate buckets. A bucket leaves when it
   is full, reaches its maximum wait, or approaches the earliest request deadline. A hard
   per-session contribution cap prevents one session from filling a batch.
5. The micro-batcher sends one length-prefixed binary batch to a stateless C++ worker. Its
   production engine executes the Open Neural Network Exchange (ONNX) model through ONNX
   Runtime; its deterministic engine exists only for protocol and failure tests. Both return
   normalized 192-dimensional embeddings.
6. Online assignment compares the embedding with profiles in the same tenant and session.
   Low-quality input becomes `spk_unknown` and cannot create or update a profile.
7. The speaker timeline joins STT word timestamps to speaker windows by maximum overlap and
   groups adjacent words assigned to the same stable identifier.

## State ownership

SQLite is the durable local reference implementation. The schema uses
`(tenant_id, session_id, speaker_id)` as its primary key. Speaker identifier allocation uses
an immediate transaction so a process restart cannot silently reuse an existing identifier.

The store contains centroid, sample count, quality exponential moving average, last-active
time, and optional display name. Inference workers contain no authoritative session state.
The same interface can be backed by Redis or another external store when multiple service
processes must share state.

## Correctness invariants

- A stale client epoch never mutates audio, centroid, or timeline state.
- A duplicate sequence number never triggers a second inference.
- All store reads and writes include `tenant_id`; identical session strings across tenants
  cannot observe one another.
- Only a finite non-zero, normalized embedding can update a centroid.
- Low-quality windows cannot create a stable speaker.
- A stable `speaker_id` is separate from `display_name`.
- Word attribution never borrows a segment from another tenant or session.
- A worker restart does not reset the session's next-speaker counter.

## Backpressure and overload

The control plane measures pending audio seconds instead of request count. The autoscaler
uses audio arrival seconds per wall-clock second, Real-Time Factor (RTF), queue P95
(95th percentile), worker utilization, and backlog. Scale-down needs several consecutive
low windows and respects a cooldown.

The current implementation returns a desired replica count; it does not mutate a
Kubernetes Deployment. An orchestrator adapter must also ensure a worker with an in-flight
batch is drained before termination. Recommended overload order:

1. keep active, high-quality speech;
2. reduce redundant overlap sampling;
3. defer low-quality windows;
4. keep long audio isolated from real-time short buckets;
5. reject new work only after explicit admission thresholds are crossed.

## Failure behavior

| Failure | Detection | Behavior |
|---|---|---|
| Duplicate or old client stream | epoch/sequence cursor | reject before frontend mutation |
| Sequence gap | gateway decision | accept, record missing range, allow retransmit control |
| Embedding backend error | batch execution exception | fail every request in that batch; no centroid mutation |
| Worker exit | request timeout/closed pipe | restart once locally; profiles remain in the store |
| State store unavailable | storage exception | fail closed for new speaker creation; do not invent an ID |
| Low-quality speech | backend quality gate | emit `spk_unknown`; defer identity learning |
| Long request | length bucket | prevent head-of-line blocking of short windows |

## Production adapters still required

- Replace or validate the baseline VAD against CueBee traces.
- Connect the desired-replica output to an orchestrator.
- Move session state from local SQLite to a shared store for multiple hosts.
- Add authenticated tenant claims at the edge and bind them to the message tenant.
- Add real trace replay, speaker-attribution accuracy, burst, and worker-kill experiments.

## Native worker boundary

Python owns packet admission, feature extraction, deadline-aware micro-batching, identity
assignment, persistence, and Speech-to-Text (STT) alignment. C++ owns the inference hot path:
bounded binary decoding, batch padding, ONNX Runtime session execution, output validation,
and length normalization. This keeps model execution stateless and independently replaceable.

The current local adapter launches one subprocess and allows one retry after an unexpected
exit. A production multi-process supervisor can route batches to several identical workers;
the wire contract is documented in [native-worker-protocol.md](native-worker-protocol.md).
