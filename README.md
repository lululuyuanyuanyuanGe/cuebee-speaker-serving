# CueBee Speaker Serving

CueBee Speaker Serving is the real-time, multi-tenant speaker diarization module used to
produce a stable answer to “who spoke when.” It combines Voice Activity Detection (VAD),
log-Mel filter-bank features, cross-session micro-batching, speaker embeddings, online
centroid assignment, and Speech-to-Text (STT) timestamp alignment.

The embedding model is intentionally not described as an Automatic Speech Recognition
(ASR) model or as an end-to-end diarization model. The production backend is designed for
`3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`, whose output is a 192-dimensional
speaker embedding. Text continues to come from a third-party streaming STT service.

## Engineering goals

- Keep inference workers stateless so a session can move between replicas.
- Share model instances across long-lived sessions with deadline-bounded micro-batches.
- Keep tenant and session speaker state isolated and recoverable.
- Preserve stable internal identifiers such as `spk_001` across worker restarts.
- Measure audio backlog seconds, queue delay, inference latency, and attribution delay
  separately.
- Keep the real Open Neural Network Exchange (ONNX) model optional so the scheduling,
  state, and correctness paths can be tested without the model artifact.

## Status

This repository is an implementation of the Speaker Serving scope only. It does not
implement CueBee's separate stateful Large Language Model (LLM) runtime work.

The implementation is built as independently testable layers. See `docs/architecture.md`
for the component boundaries and `python -m unittest discover -s tests -v` for the
correctness suite.

## Evidence discipline

Benchmark output from this repository is labeled with its workload and backend. Synthetic
or model-only measurements must not be presented as production end-to-end latency. The
64-live-session figure in the project brief remains a capacity estimate until a matching
trace replay has been run.

