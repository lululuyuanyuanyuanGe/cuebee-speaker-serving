"""Deadline-bounded, length-bucketed cross-session micro-batching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import BatcherConfig
from ..domain import EmbeddingRequest, EmbeddingResult, FeatureChunk
from .backend import EmbeddingBackend


@dataclass(frozen=True)
class BatchObservation:
    bucket: int
    batch_size: int
    audio_seconds: float
    oldest_queue_delay_ms: float
    inference_ms: float


@dataclass
class _QueuedRequest:
    request: EmbeddingRequest
    future: "asyncio.Future[EmbeddingResult]"
    ordinal: int

    @property
    def session_key(self) -> Tuple[str, str]:
        chunk = self.request.chunk
        return (chunk.tenant_id, chunk.session_id)


class CrossSessionMicroBatcher:
    """Aggregate requests from different sessions without violating queue deadlines."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        config: BatcherConfig,
        worker_id: str = "worker-local",
        observer: Optional[Callable[[BatchObservation], None]] = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._worker_id = worker_id
        self._observer = observer
        self._queues: Dict[int, List[_QueuedRequest]] = {}
        self._condition = asyncio.Condition()
        self._runner: Optional["asyncio.Task[None]"] = None
        self._closing = False
        self._ordinal = 0

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("batcher is closing")
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name="cuebee-micro-batcher")

    async def submit(
        self, chunk: FeatureChunk, deadline_ms: Optional[int] = None
    ) -> EmbeddingResult:
        await self.start()
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[EmbeddingResult]" = loop.create_future()
        deadline_delay = (
            self._config.default_deadline_ms if deadline_ms is None else deadline_ms
        )
        if deadline_delay <= 0:
            raise ValueError("deadline_ms must be positive")
        now = time.monotonic()
        request = EmbeddingRequest(
            chunk=chunk,
            deadline_at=now + deadline_delay / 1_000.0,
            enqueued_at=now,
        )
        bucket = self._bucket_for_frames(chunk.features.shape[0])
        async with self._condition:
            if self._closing:
                raise RuntimeError("batcher is closing")
            queued = _QueuedRequest(request=request, future=future, ordinal=self._ordinal)
            self._ordinal += 1
            self._queues.setdefault(bucket, []).append(queued)
            self._condition.notify_all()
        return await future

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._runner is not None:
            await self._runner
            self._runner = None

    async def backlog(self) -> Tuple[int, float]:
        async with self._condition:
            items = [item for queue in self._queues.values() for item in queue]
            return len(items), sum(item.request.chunk.audio_seconds for item in items)

    async def __aenter__(self) -> "CrossSessionMicroBatcher":
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    def _bucket_for_frames(self, frame_count: int) -> int:
        for index, upper_bound in enumerate(self._config.bucket_boundaries_frames):
            if frame_count <= upper_bound:
                return index
        return len(self._config.bucket_boundaries_frames)

    async def _run(self) -> None:
        while True:
            batch: List[_QueuedRequest]
            bucket: int
            async with self._condition:
                while not self._has_pending() and not self._closing:
                    await self._condition.wait()
                if self._closing and not self._has_pending():
                    return

                selection = self._select_ready_bucket(time.monotonic())
                if selection is None:
                    wake_at = self._next_wake_at()
                    timeout = max(0.0, wake_at - time.monotonic())
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
                    continue
                bucket = selection
                batch = self._take_batch(bucket)

            await self._execute(bucket, batch)

    def _has_pending(self) -> bool:
        return any(self._queues.values())

    def _flush_at(self, queue: List[_QueuedRequest]) -> float:
        oldest = min(item.request.enqueued_at for item in queue)
        earliest_deadline = min(item.request.deadline_at for item in queue)
        return min(oldest + self._config.max_wait_ms / 1_000.0, earliest_deadline)

    def _select_ready_bucket(self, now: float) -> Optional[int]:
        ready = []
        for bucket, queue in self._queues.items():
            if not queue:
                continue
            if (
                self._closing
                or len(queue) >= self._config.max_batch_size
                or now >= self._flush_at(queue)
            ):
                earliest_deadline = min(item.request.deadline_at for item in queue)
                ready.append((earliest_deadline, self._flush_at(queue), bucket))
        if not ready:
            return None
        return min(ready)[2]

    def _next_wake_at(self) -> float:
        return min(self._flush_at(queue) for queue in self._queues.values() if queue)

    def _take_batch(self, bucket: int) -> List[_QueuedRequest]:
        queue = self._queues[bucket]
        ordered = sorted(
            queue,
            key=lambda item: (
                item.request.deadline_at,
                item.request.enqueued_at,
                item.ordinal,
            ),
        )
        selected: List[_QueuedRequest] = []
        per_session: Dict[Tuple[str, str], int] = {}
        for item in ordered:
            count = per_session.get(item.session_key, 0)
            if count >= self._config.per_session_batch_limit:
                continue
            selected.append(item)
            per_session[item.session_key] = count + 1
            if len(selected) >= self._config.max_batch_size:
                break

        if not selected:
            selected.append(ordered[0])
        selected_ids = {id(item) for item in selected}
        self._queues[bucket] = [item for item in queue if id(item) not in selected_ids]
        if not self._queues[bucket]:
            del self._queues[bucket]
        return selected

    async def _execute(self, bucket: int, batch: List[_QueuedRequest]) -> None:
        started_at = time.monotonic()
        try:
            embeddings, qualities = await asyncio.to_thread(
                self._backend.infer_batch,
                [item.request.chunk.features for item in batch],
            )
            completed_at = time.monotonic()
            embeddings = np.asarray(embeddings, dtype=np.float32)
            qualities = np.asarray(qualities, dtype=np.float32)
            expected_shape = (len(batch), self._backend.embedding_dimension)
            if embeddings.shape != expected_shape or qualities.shape != (len(batch),):
                raise ValueError(
                    "backend returned invalid shapes: "
                    f"embeddings={embeddings.shape}, qualities={qualities.shape}"
                )
            inference_ms = (completed_at - started_at) * 1_000.0
            for index, item in enumerate(batch):
                if item.future.cancelled():
                    continue
                item.future.set_result(
                    EmbeddingResult(
                        chunk=item.request.chunk,
                        embedding=embeddings[index],
                        quality=float(np.clip(qualities[index], 0.0, 1.0)),
                        worker_id=self._worker_id,
                        batch_size=len(batch),
                        queue_delay_ms=(started_at - item.request.enqueued_at) * 1_000.0,
                        inference_ms=inference_ms,
                    )
                )
            if self._observer is not None:
                observation = BatchObservation(
                    bucket=bucket,
                    batch_size=len(batch),
                    audio_seconds=sum(item.request.chunk.audio_seconds for item in batch),
                    oldest_queue_delay_ms=max(
                        (started_at - item.request.enqueued_at) * 1_000.0 for item in batch
                    ),
                    inference_ms=inference_ms,
                )
                try:
                    self._observer(observation)
                except Exception:
                    pass
        except Exception as error:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)

