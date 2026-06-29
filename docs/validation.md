# Validation runbook

## Correctness suite

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers gateway idempotency, tenant isolation, audio validation, feature shape,
length-bucket separation, cross-session batch fill, low-quality rejection, stable identity
after SQLite restart, word-level alignment, autoscaling hysteresis, server dispatch, and an
audio-to-attributed-transcript integration path.

## Functional demo

```bash
PYTHONPATH=src python -m cuebee_speaker.demo
```

Expected identity pattern: `spk_001`, `spk_002`, `spk_001`. This uses synthetic tones and
the deterministic development backend. It proves event flow and state transitions, not
speaker-attribution accuracy.

## Scheduling load test

```bash
PYTHONPATH=src python benchmarks/loadgen.py --sessions 32 --requests-per-session 10
```

Run at least five repetitions after warmup. Retain the command, environment, raw JSON,
processor model, Python/NumPy versions, and model checksum. Do not compare the development
backend numbers with ONNX Runtime results.

## Real model experiment matrix

| Group | Configuration | Question |
|---|---|---|
| S0 | synchronous Batch 1 | model-only latency baseline |
| S1 | shared first-in-first-out batches | shared-worker throughput |
| S2 | VAD + length buckets + deadline departure | padding and tail latency |
| S3 | S2 + external state + autoscaler adapter | recovery and elasticity |

For each group record P50/P95/P99 queue delay, inference latency, end-to-end attribution
delay, audio seconds per second, RTF, batch fill, and accuracy on labeled traces.

## Failure drill

1. Start two workers against one shared state-store adapter.
2. Establish at least two speakers in one session and record their stable identifiers.
3. Keep the load generator at the planned nominal rate.
4. Terminate one worker while it has an in-flight batch.
5. Retry unacknowledged chunks on the remaining worker with the same epoch and sequence.
6. Verify that no duplicate timeline window appears and that both speaker identifiers are
   unchanged.
7. Record queue P99 (99th percentile), attribution P99, retry count, and recovery time.

Local SQLite proves process restart on one host. A multi-host drill requires a shared store;
it must not be claimed from the SQLite-only test.

