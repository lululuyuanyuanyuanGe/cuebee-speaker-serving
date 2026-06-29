"""Synthetic end-to-end demonstration of the serving and alignment path."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import asdict

import numpy as np

from .audio import float32_to_pcm16le
from .config import AppConfig, AssignmentConfig, AudioConfig, BatcherConfig
from .domain import AudioChunk, STTSegment, STTWord
from .factory import build_runtime


async def run_demo() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        config = AppConfig(
            audio=AudioConfig(window_ms=1_500, hop_ms=1_500),
            batcher=BatcherConfig(max_batch_size=4, max_wait_ms=5),
            assignment=AssignmentConfig(cosine_threshold=0.98, min_quality=0.20),
            state_db_path=os.path.join(directory, "speaker-state.sqlite3"),
        )
        runtime = build_runtime(config)
        segments = []
        frequencies = (180.0, 380.0, 180.0)
        try:
            for sequence in range(45):
                speaker_window = sequence // 15
                packet_in_window = sequence % 15
                local_indices = (
                    np.arange(1_600, dtype=np.float32) + packet_in_window * 1_600
                )
                local_time = local_indices / 16_000.0
                frequency = frequencies[speaker_window]
                samples = (
                    0.24
                    * (1.0 + 0.4 * np.sin(2 * np.pi * 3 * local_time))
                    * np.sin(2 * np.pi * frequency * local_time)
                ).astype(np.float32)
                result = await runtime.pipeline.ingest_audio(
                    AudioChunk(
                        tenant_id="demo-tenant",
                        session_id="demo-session",
                        client_epoch=1,
                        seq_no=sequence,
                        start_ms=sequence * 100,
                        end_ms=(sequence + 1) * 100,
                        pcm=float32_to_pcm16le(samples),
                    )
                )
                segments.extend(result.speaker_segments)

            if len({item.speaker_id for item in segments}) != 2:
                raise RuntimeError("synthetic demonstration did not form two speaker clusters")
            first_id, second_id = segments[0].speaker_id, segments[1].speaker_id
            await runtime.pipeline.rename_speaker(
                "demo-tenant", "demo-session", first_id, "Alice"
            )
            await runtime.pipeline.rename_speaker(
                "demo-tenant", "demo-session", second_id, "Bob"
            )

            stt = STTSegment(
                tenant_id="demo-tenant",
                session_id="demo-session",
                segment_id="stt-final-1",
                revision=1,
                text="Alice reviews the plan Bob raises a risk Alice confirms the action",
                start_ms=0,
                end_ms=4_500,
                is_final=True,
                words=(
                    STTWord("Alice", 0, 250),
                    STTWord("reviews", 250, 650),
                    STTWord("the", 650, 800),
                    STTWord("plan", 800, 1_300),
                    STTWord("Bob", 1_550, 1_800),
                    STTWord("raises", 1_800, 2_200),
                    STTWord("a", 2_200, 2_300),
                    STTWord("risk", 2_300, 2_800),
                    STTWord("Alice", 3_050, 3_300),
                    STTWord("confirms", 3_300, 3_750),
                    STTWord("the", 3_750, 3_900),
                    STTWord("action", 3_900, 4_350),
                ),
            )
            spans = runtime.pipeline.align_stt(stt)
            profiles = await runtime.pipeline.assigner.profiles(
                "demo-tenant", "demo-session"
            )
            return {
                "evidence": "synthetic functional demo; not an accuracy benchmark",
                "backend": runtime.backend_name,
                "speaker_segments": [asdict(item) for item in segments],
                "attributed_transcript": [asdict(item) for item in spans],
                "profiles": [
                    {
                        "speaker_id": item.speaker_id,
                        "display_name": item.display_name,
                        "sample_count": item.sample_count,
                    }
                    for item in profiles
                ],
                "metrics": runtime.metrics.snapshot(),
            }
        finally:
            await runtime.close()


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

