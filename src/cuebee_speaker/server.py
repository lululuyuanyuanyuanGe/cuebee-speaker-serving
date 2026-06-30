"""Newline-delimited JavaScript Object Notation (JSON) streaming service."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from dataclasses import asdict
from typing import Any, Dict, Optional

from .autoscaler import ScalingSample
from .config import AppConfig
from .domain import AudioChunk, STTSegment, STTWord
from .factory import ServiceRuntime, build_runtime


class SpeakerTCPServer:
    """A zero-framework streaming gateway suitable for a trusted internal network.

    Authentication and Transport Layer Security (TLS) termination belong at the edge
    proxy. Every data-plane message still requires an authenticated tenant identifier.
    """

    def __init__(self, runtime: ServiceRuntime, max_line_bytes: int = 4 * 1024 * 1024) -> None:
        self.runtime = runtime
        self.max_line_bytes = max_line_bytes

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                if len(line) > self.max_line_bytes:
                    await self._write(writer, {"ok": False, "error": "message_too_large"})
                    break
                try:
                    message = json.loads(line)
                    response = await self.dispatch(message)
                except (ValueError, TypeError, KeyError) as error:
                    response = {"ok": False, "error": str(error)}
                await self._write(writer, response)
        finally:
            writer.close()
            await writer.wait_closed()
            self.runtime.metrics.increment("connections_closed")
            del peer

    async def dispatch(self, message: Dict[str, Any]) -> Dict[str, Any]:
        event_type = message.get("type")
        if event_type == "health":
            return {"ok": True, "backend": self.runtime.backend_name}
        if event_type == "metrics":
            return {"ok": True, "metrics": self.runtime.metrics.snapshot()}
        if event_type == "audio":
            payload = base64.b64decode(message["pcm_base64"], validate=True)
            chunk = AudioChunk(
                tenant_id=message["tenant_id"],
                session_id=message["session_id"],
                client_epoch=int(message["client_epoch"]),
                seq_no=int(message["seq_no"]),
                start_ms=int(message["start_ms"]),
                end_ms=int(message["end_ms"]),
                pcm=payload,
                sample_rate=int(message.get("sample_rate", 16_000)),
                channels=int(message.get("channels", 1)),
                is_final=bool(message.get("is_final", False)),
            )
            result = await self.runtime.pipeline.ingest_audio(chunk)
            return {
                "ok": True,
                "gateway": result.gateway.status.value,
                "expected_next_seq": result.gateway.expected_next_seq,
                "missing_range": result.gateway.missing_range,
                "speaker_segments": [asdict(item) for item in result.speaker_segments],
            }
        if event_type == "stt":
            words = tuple(
                STTWord(
                    text=item["text"],
                    start_ms=int(item["start_ms"]),
                    end_ms=int(item["end_ms"]),
                )
                for item in message.get("words", ())
            )
            event = STTSegment(
                tenant_id=message["tenant_id"],
                session_id=message["session_id"],
                segment_id=message["segment_id"],
                revision=int(message["revision"]),
                text=message["text"],
                start_ms=int(message["start_ms"]),
                end_ms=int(message["end_ms"]),
                is_final=bool(message["is_final"]),
                words=words,
            )
            spans = self.runtime.pipeline.align_stt(event)
            return {"ok": True, "attributed_spans": [asdict(item) for item in spans]}
        if event_type == "profiles":
            profiles = await self.runtime.pipeline.assigner.profiles(
                message["tenant_id"], message["session_id"]
            )
            return {
                "ok": True,
                "profiles": [
                    {
                        "speaker_id": item.speaker_id,
                        "display_name": item.display_name,
                        "sample_count": item.sample_count,
                        "quality_ema": item.quality_ema,
                        "last_active_ms": item.last_active_ms,
                        "embedding_dimension": int(item.centroid.size),
                    }
                    for item in profiles
                ],
            }
        if event_type == "rename":
            renamed = await self.runtime.pipeline.rename_speaker(
                message["tenant_id"],
                message["session_id"],
                message["speaker_id"],
                message["display_name"],
            )
            return {"ok": renamed}
        if event_type == "flush":
            segments = await self.runtime.pipeline.flush_session(
                message["tenant_id"], message["session_id"]
            )
            return {"ok": True, "speaker_segments": [asdict(item) for item in segments]}
        if event_type == "autoscale":
            sample = ScalingSample(
                audio_arrival_seconds_per_second=float(
                    message["audio_arrival_seconds_per_second"]
                ),
                real_time_factor=float(message["real_time_factor"]),
                backlog_seconds=float(message["backlog_seconds"]),
                queue_p95_ms=float(message["queue_p95_ms"]),
                worker_utilization=float(message["worker_utilization"]),
                current_replicas=int(message["current_replicas"]),
                observed_at=float(message.get("observed_at", 0.0)),
            )
            return {"ok": True, "decision": asdict(self.runtime.autoscaler.recommend(sample))}
        raise ValueError(f"unsupported message type: {event_type!r}")

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, response: Dict[str, Any]) -> None:
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()


async def _serve(args: argparse.Namespace) -> None:
    config = AppConfig(
        model_path=args.model,
        native_worker_path=args.native_worker,
        native_worker_backend=args.native_backend,
        native_worker_timeout_seconds=args.worker_timeout,
        native_worker_intra_op_threads=args.intra_op_threads,
        state_db_path=args.state_db,
    )
    runtime = build_runtime(config)
    gateway = SpeakerTCPServer(runtime)
    server = await asyncio.start_server(
        gateway.handle_client, args.host, args.port, limit=gateway.max_line_bytes + 1
    )
    addresses = ", ".join(str(socket.getsockname()) for socket in server.sockets or ())
    print(f"CueBee speaker server listening on {addresses}; backend={runtime.backend_name}")
    try:
        async with server:
            await server.serve_forever()
    finally:
        await runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CueBee real-time speaker serving")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--model", help="path to the ERes2Net ONNX artifact")
    parser.add_argument("--native-worker", help="path to the C++ embedding worker")
    parser.add_argument(
        "--native-backend",
        choices=("auto", "deterministic", "onnx"),
        default="auto",
        help="native engine; auto selects ONNX when --model is present",
    )
    parser.add_argument("--worker-timeout", default=5.0, type=float)
    parser.add_argument("--intra-op-threads", default=0, type=int)
    parser.add_argument("--state-db", default="runtime/speaker-state.sqlite3")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
