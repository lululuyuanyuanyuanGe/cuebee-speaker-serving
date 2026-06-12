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
    if config.model_path:
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

