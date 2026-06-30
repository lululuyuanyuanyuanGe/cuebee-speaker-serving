"""Client for the length-prefixed CueBee native embedding worker protocol."""

from __future__ import annotations

import collections
import os
import queue
import struct
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


_REQUEST_HEADER = struct.Struct("<4sHHQIIII")
_RESPONSE_HEADER = struct.Struct("<4sHHQIIII")
_U32 = struct.Struct("<I")
_PROTOCOL_VERSION = 1
_OP_INFER = 1
_OP_PING = 2
_OP_SHUTDOWN = 3
_STATUS_OK = 0


class NativeWorkerError(RuntimeError):
    """Base error raised by the native worker adapter."""


class NativeWorkerUnavailable(NativeWorkerError):
    """The subprocess or its transport stopped responding."""


class NativeWorkerExecutionError(NativeWorkerError):
    """The worker accepted a request but its inference engine rejected it."""


@dataclass(frozen=True)
class _WorkerResponse:
    request_id: int
    status: int
    error: str
    embeddings: np.ndarray
    qualities: np.ndarray


def _read_exact(stream: object, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)  # type: ignore[attr-defined]
        if not chunk:
            raise NativeWorkerUnavailable("native worker closed its response stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NativeWorkerBackend:
    """Persistent C++ worker process implementing the embedding backend contract."""

    embedding_dimension = 192

    def __init__(
        self,
        worker_path: str,
        model_path: Optional[str] = None,
        backend: str = "deterministic",
        timeout_seconds: float = 5.0,
        intra_op_threads: int = 0,
        max_batch_size: int = 64,
        max_frames: int = 2_000,
        restart_attempts: int = 1,
    ) -> None:
        path = Path(worker_path)
        if not path.is_file():
            raise FileNotFoundError(f"native worker not found: {worker_path}")
        if not os.access(str(path), os.X_OK):
            raise PermissionError(f"native worker is not executable: {worker_path}")
        if backend not in {"deterministic", "onnx"}:
            raise ValueError("native backend must be 'deterministic' or 'onnx'")
        if backend == "onnx" and not model_path:
            raise ValueError("model_path is required for the native ONNX backend")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_batch_size <= 0 or max_frames <= 0:
            raise ValueError("worker limits must be positive")
        if restart_attempts < 0:
            raise ValueError("restart_attempts cannot be negative")

        self._worker_path = str(path.resolve())
        self._model_path = str(Path(model_path).resolve()) if model_path else None
        self._backend = backend
        self._timeout_seconds = timeout_seconds
        self._intra_op_threads = intra_op_threads
        self._max_batch_size = max_batch_size
        self._max_frames = max_frames
        self._restart_attempts = restart_attempts
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._responses: "queue.Queue[object]" = queue.Queue()
        self._next_request_id = 1
        self._stderr_lines: Deque[str] = collections.deque(maxlen=20)
        self._closed = False

        with self._lock:
            self._start_process()
            try:
                self._exchange(_OP_PING, ())
            except Exception:
                self._stop_process()
                raise

    @property
    def worker_pid(self) -> Optional[int]:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def infer_batch(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not features:
            return (
                np.empty((0, self.embedding_dimension), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        matrices = self._validate_features(features)
        with self._lock:
            if self._closed:
                raise NativeWorkerUnavailable("native worker backend is closed")
            for attempt in range(self._restart_attempts + 1):
                try:
                    response = self._exchange(_OP_INFER, matrices)
                    return response.embeddings, response.qualities
                except NativeWorkerUnavailable:
                    self._stop_process()
                    if attempt >= self._restart_attempts:
                        raise
                    self._start_process()
        raise AssertionError("native worker retry loop did not return")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._exchange(_OP_SHUTDOWN, ())
                except NativeWorkerError:
                    pass
            self._stop_process()

    def __enter__(self) -> "NativeWorkerBackend":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _validate_features(self, features: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
        if len(features) > self._max_batch_size:
            raise ValueError("batch exceeds native worker maximum")
        matrices = []
        for item in features:
            matrix = np.asarray(item, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 80:
                raise ValueError("native worker expects non-empty [time, 80] features")
            if matrix.shape[0] > self._max_frames:
                raise ValueError("feature sequence exceeds native worker frame maximum")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("features must contain only finite values")
            matrices.append(np.ascontiguousarray(matrix, dtype="<f4"))
        return tuple(matrices)

    def _command(self) -> List[str]:
        command = [
            self._worker_path,
            "--backend",
            self._backend,
            "--max-batch-size",
            str(self._max_batch_size),
            "--max-frames",
            str(self._max_frames),
        ]
        if self._model_path:
            command.extend(("--model", self._model_path))
        if self._intra_op_threads > 0:
            command.extend(("--intra-op-threads", str(self._intra_op_threads)))
        return command

    def _start_process(self) -> None:
        self._responses = queue.Queue()
        self._stderr_lines.clear()
        try:
            process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as error:
            raise NativeWorkerUnavailable(f"failed to start native worker: {error}") from error
        self._process = process
        threading.Thread(
            target=self._reader_loop,
            args=(process, self._responses),
            name="cuebee-native-worker-reader",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stderr_loop,
            args=(process,),
            name="cuebee-native-worker-stderr",
            daemon=True,
        ).start()

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _exchange(
        self, operation: int, features: Sequence[np.ndarray]
    ) -> _WorkerResponse:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            detail = "; ".join(self._stderr_lines)
            suffix = f": {detail}" if detail else ""
            raise NativeWorkerUnavailable(f"native worker is not running{suffix}")

        request_id = self._next_request_id
        self._next_request_id += 1
        payload_parts = []
        for matrix in features:
            payload_parts.append(_U32.pack(matrix.shape[0]))
            payload_parts.append(matrix.tobytes(order="C"))
        payload = b"".join(payload_parts)
        header = _REQUEST_HEADER.pack(
            b"CBSP",
            _PROTOCOL_VERSION,
            operation,
            request_id,
            len(features),
            80,
            len(payload),
            0,
        )
        try:
            process.stdin.write(header)
            if payload:
                process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise NativeWorkerUnavailable(
                f"failed to write native worker request: {error}"
            ) from error

        try:
            message = self._responses.get(timeout=self._timeout_seconds)
        except queue.Empty as error:
            raise NativeWorkerUnavailable(
                f"native worker timed out after {self._timeout_seconds:.3f} seconds"
            ) from error
        if isinstance(message, Exception):
            raise NativeWorkerUnavailable(str(message)) from message
        response = message
        if not isinstance(response, _WorkerResponse):
            raise NativeWorkerUnavailable("native worker reader returned an invalid message")
        if response.request_id != request_id:
            raise NativeWorkerUnavailable(
                f"native worker response mismatch: expected {request_id}, got {response.request_id}"
            )
        if response.status != _STATUS_OK:
            raise NativeWorkerExecutionError(
                response.error or f"native worker failed with status {response.status}"
            )
        return response

    @staticmethod
    def _reader_loop(
        process: subprocess.Popen[bytes], responses: "queue.Queue[object]"
    ) -> None:
        try:
            if process.stdout is None:
                raise NativeWorkerUnavailable("native worker has no response stream")
            while True:
                header = _read_exact(process.stdout, _RESPONSE_HEADER.size)
                (
                    magic,
                    version,
                    status,
                    request_id,
                    batch_size,
                    dimension,
                    payload_bytes,
                    error_bytes,
                ) = _RESPONSE_HEADER.unpack(header)
                if magic != b"CBSR" or version != _PROTOCOL_VERSION:
                    raise NativeWorkerUnavailable(
                        "native worker returned an invalid response header"
                    )
                if payload_bytes > 256 * 1024 * 1024 or error_bytes > payload_bytes:
                    raise NativeWorkerUnavailable("native worker response exceeds protocol limits")
                payload = _read_exact(process.stdout, payload_bytes) if payload_bytes else b""
                error_text = payload[:error_bytes].decode("utf-8", errors="replace")
                result_payload = payload[error_bytes:]
                if status == _STATUS_OK:
                    if batch_size == 0:
                        if dimension != 0 or result_payload:
                            raise NativeWorkerUnavailable("invalid empty native worker response")
                        embeddings = np.empty((0, 192), dtype=np.float32)
                        qualities = np.empty((0,), dtype=np.float32)
                    else:
                        if dimension != 192:
                            raise NativeWorkerUnavailable(
                                f"native worker returned embedding dimension {dimension}"
                            )
                        expected_bytes = batch_size * (dimension + 1) * 4
                        if len(result_payload) != expected_bytes:
                            raise NativeWorkerUnavailable(
                                "native worker response has invalid length"
                            )
                        records = np.frombuffer(result_payload, dtype="<f4").reshape(
                            batch_size, dimension + 1
                        )
                        qualities = records[:, 0].astype(np.float32, copy=True)
                        embeddings = records[:, 1:].astype(np.float32, copy=True)
                else:
                    embeddings = np.empty((0, 192), dtype=np.float32)
                    qualities = np.empty((0,), dtype=np.float32)
                responses.put(
                    _WorkerResponse(
                        request_id=request_id,
                        status=status,
                        error=error_text,
                        embeddings=embeddings,
                        qualities=qualities,
                    )
                )
        except Exception as error:
            responses.put(error)

    def _stderr_loop(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for raw_line in iter(stream.readline, b""):
                self._stderr_lines.append(raw_line.decode("utf-8", errors="replace").strip())
        except (OSError, ValueError):
            pass


__all__ = [
    "NativeWorkerBackend",
    "NativeWorkerError",
    "NativeWorkerExecutionError",
    "NativeWorkerUnavailable",
]
