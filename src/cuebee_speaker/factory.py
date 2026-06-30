"""Runtime assembly for local development and deployable service entry points."""

from __future__ import annotations

from dataclasses import dataclass

from .assignment import OnlineSpeakerAssigner
from .audio import StreamingAudioFrontend
from .autoscaler import AudioBacklogAutoscaler
from .config import AppConfig
from .gateway import EventGateway
from .inference import (
    CrossSessionMicroBatcher,
    DeterministicEmbeddingBackend,
    NativeWorkerBackend,
    ONNXEmbeddingBackend,
)
from .metrics import MetricRegistry
from .pipeline import DiarizationPipeline
from .state import SQLiteSpeakerStore


@dataclass
class ServiceRuntime:
    pipeline: DiarizationPipeline
    metrics: MetricRegistry
    autoscaler: AudioBacklogAutoscaler
    backend_name: str

    async def close(self) -> None:
        await self.pipeline.close()


def build_runtime(config: AppConfig) -> ServiceRuntime:
    metrics = MetricRegistry()
    if config.native_worker_path:
        native_backend = config.native_worker_backend
        if native_backend == "auto":
            native_backend = "onnx" if config.model_path else "deterministic"
        backend = NativeWorkerBackend(
            worker_path=config.native_worker_path,
            model_path=config.model_path,
            backend=native_backend,
            timeout_seconds=config.native_worker_timeout_seconds,
            intra_op_threads=config.native_worker_intra_op_threads,
            max_batch_size=config.batcher.max_batch_size,
            restart_attempts=config.native_worker_restart_attempts,
        )
        backend_name = (
            "native-cpp-onnxruntime-eres2net"
            if native_backend == "onnx"
            else "native-cpp-deterministic-development"
        )
    elif config.model_path:
        backend = ONNXEmbeddingBackend(config.model_path)
        backend_name = "onnxruntime-eres2net"
    else:
        backend = DeterministicEmbeddingBackend()
        backend_name = "deterministic-development-backend"
    batcher = CrossSessionMicroBatcher(
        backend=backend,
        config=config.batcher,
        observer=metrics.observe_batch,
    )
    store = SQLiteSpeakerStore(config.state_db_path)
    pipeline = DiarizationPipeline(
        gateway=EventGateway(),
        frontend=StreamingAudioFrontend(config.audio),
        batcher=batcher,
        assigner=OnlineSpeakerAssigner(store, config.assignment),
        metrics=metrics,
    )
    return ServiceRuntime(
        pipeline=pipeline,
        metrics=metrics,
        autoscaler=AudioBacklogAutoscaler(config.autoscaler),
        backend_name=backend_name,
    )
