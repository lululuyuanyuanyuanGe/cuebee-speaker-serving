"""Tenant-isolated in-memory and SQLite speaker profile stores."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..domain import SpeakerProfile


class SpeakerStore(Protocol):
    def list_profiles(self, tenant_id: str, session_id: str) -> List[SpeakerProfile]:
        ...

    def create_profile(
        self,
        tenant_id: str,
        session_id: str,
        centroid: np.ndarray,
        quality: float,
        last_active_ms: int,
    ) -> SpeakerProfile:
        ...

    def update_profile(self, profile: SpeakerProfile) -> None:
        ...

    def rename_speaker(
        self, tenant_id: str, session_id: str, speaker_id: str, display_name: Optional[str]
    ) -> bool:
        ...

    def delete_session(self, tenant_id: str, session_id: str) -> None:
        ...

    def close(self) -> None:
        ...


def _copy_profile(profile: SpeakerProfile) -> SpeakerProfile:
    return SpeakerProfile(
        tenant_id=profile.tenant_id,
        session_id=profile.session_id,
        speaker_id=profile.speaker_id,
        centroid=np.array(profile.centroid, dtype=np.float32, copy=True),
        sample_count=profile.sample_count,
        quality_ema=profile.quality_ema,
        last_active_ms=profile.last_active_ms,
        display_name=profile.display_name,
    )


class InMemorySpeakerStore:
    def __init__(self) -> None:
        self._profiles: Dict[Tuple[str, str, str], SpeakerProfile] = {}
        self._next_ids: Dict[Tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def list_profiles(self, tenant_id: str, session_id: str) -> List[SpeakerProfile]:
        with self._lock:
            result = [
                _copy_profile(profile)
                for key, profile in self._profiles.items()
                if key[:2] == (tenant_id, session_id)
            ]
        return sorted(result, key=lambda profile: profile.speaker_id)

    def create_profile(
        self,
        tenant_id: str,
        session_id: str,
        centroid: np.ndarray,
        quality: float,
        last_active_ms: int,
    ) -> SpeakerProfile:
        with self._lock:
            session_key = (tenant_id, session_id)
            next_id = self._next_ids.get(session_key, 1)
            self._next_ids[session_key] = next_id + 1
            profile = SpeakerProfile(
                tenant_id=tenant_id,
                session_id=session_id,
                speaker_id=f"spk_{next_id:03d}",
                centroid=np.asarray(centroid, dtype=np.float32).copy(),
                sample_count=1,
                quality_ema=quality,
                last_active_ms=last_active_ms,
            )
            self._profiles[(tenant_id, session_id, profile.speaker_id)] = profile
            return _copy_profile(profile)

    def update_profile(self, profile: SpeakerProfile) -> None:
        with self._lock:
            key = (profile.tenant_id, profile.session_id, profile.speaker_id)
            if key not in self._profiles:
                raise KeyError(f"speaker profile not found: {key}")
            self._profiles[key] = _copy_profile(profile)

    def rename_speaker(
        self, tenant_id: str, session_id: str, speaker_id: str, display_name: Optional[str]
    ) -> bool:
        with self._lock:
            key = (tenant_id, session_id, speaker_id)
            profile = self._profiles.get(key)
            if profile is None:
                return False
            profile.display_name = display_name
            return True

    def delete_session(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            keys = [key for key in self._profiles if key[:2] == (tenant_id, session_id)]
            for key in keys:
                del self._profiles[key]
            self._next_ids.pop((tenant_id, session_id), None)

    def close(self) -> None:
        return None


class SQLiteSpeakerStore:
    """Durable local state used to prove worker restart and identity continuity."""

    def __init__(self, path: str) -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(database_path), timeout=30.0, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS speaker_profiles (
                tenant_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                centroid BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                quality_ema REAL NOT NULL,
                last_active_ms INTEGER NOT NULL,
                display_name TEXT,
                PRIMARY KEY (tenant_id, session_id, speaker_id)
            );
            CREATE TABLE IF NOT EXISTS session_counters (
                tenant_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                next_speaker INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_speaker_last_active
                ON speaker_profiles (tenant_id, last_active_ms);
            """
        )

    def list_profiles(self, tenant_id: str, session_id: str) -> List[SpeakerProfile]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT tenant_id, session_id, speaker_id, centroid, dimension,
                       sample_count, quality_ema, last_active_ms, display_name
                FROM speaker_profiles
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY speaker_id
                """,
                (tenant_id, session_id),
            ).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def create_profile(
        self,
        tenant_id: str,
        session_id: str,
        centroid: np.ndarray,
        quality: float,
        last_active_ms: int,
    ) -> SpeakerProfile:
        vector = np.asarray(centroid, dtype="<f4").reshape(-1)
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT next_speaker FROM session_counters
                    WHERE tenant_id = ? AND session_id = ?
                    """,
                    (tenant_id, session_id),
                ).fetchone()
                if row is None:
                    existing = self._connection.execute(
                        """
                        SELECT speaker_id FROM speaker_profiles
                        WHERE tenant_id = ? AND session_id = ?
                        """,
                        (tenant_id, session_id),
                    ).fetchall()
                    suffixes = [
                        int(item["speaker_id"].split("_")[-1])
                        for item in existing
                        if item["speaker_id"].split("_")[-1].isdigit()
                    ]
                    next_id = max(suffixes, default=0) + 1
                    self._connection.execute(
                        """
                        INSERT INTO session_counters (tenant_id, session_id, next_speaker)
                        VALUES (?, ?, ?)
                        """,
                        (tenant_id, session_id, next_id + 1),
                    )
                else:
                    next_id = int(row["next_speaker"])
                    self._connection.execute(
                        """
                        UPDATE session_counters SET next_speaker = ?
                        WHERE tenant_id = ? AND session_id = ?
                        """,
                        (next_id + 1, tenant_id, session_id),
                    )
                speaker_id = f"spk_{next_id:03d}"
                self._connection.execute(
                    """
                    INSERT INTO speaker_profiles (
                        tenant_id, session_id, speaker_id, centroid, dimension,
                        sample_count, quality_ema, last_active_ms, display_name
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, NULL)
                    """,
                    (
                        tenant_id,
                        session_id,
                        speaker_id,
                        vector.tobytes(),
                        vector.size,
                        quality,
                        last_active_ms,
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return SpeakerProfile(
            tenant_id=tenant_id,
            session_id=session_id,
            speaker_id=speaker_id,
            centroid=vector.astype(np.float32),
            sample_count=1,
            quality_ema=quality,
            last_active_ms=last_active_ms,
        )

    def update_profile(self, profile: SpeakerProfile) -> None:
        vector = np.asarray(profile.centroid, dtype="<f4").reshape(-1)
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE speaker_profiles
                SET centroid = ?, dimension = ?, sample_count = ?, quality_ema = ?,
                    last_active_ms = ?, display_name = ?
                WHERE tenant_id = ? AND session_id = ? AND speaker_id = ?
                """,
                (
                    vector.tobytes(),
                    vector.size,
                    profile.sample_count,
                    profile.quality_ema,
                    profile.last_active_ms,
                    profile.display_name,
                    profile.tenant_id,
                    profile.session_id,
                    profile.speaker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"speaker profile not found: {profile.speaker_id}")

    def rename_speaker(
        self, tenant_id: str, session_id: str, speaker_id: str, display_name: Optional[str]
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE speaker_profiles SET display_name = ?
                WHERE tenant_id = ? AND session_id = ? AND speaker_id = ?
                """,
                (display_name, tenant_id, session_id, speaker_id),
            )
            return cursor.rowcount == 1

    def delete_session(self, tenant_id: str, session_id: str) -> None:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "DELETE FROM speaker_profiles WHERE tenant_id = ? AND session_id = ?",
                    (tenant_id, session_id),
                )
                self._connection.execute(
                    "DELETE FROM session_counters WHERE tenant_id = ? AND session_id = ?",
                    (tenant_id, session_id),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> SpeakerProfile:
        dimension = int(row["dimension"])
        centroid = np.frombuffer(row["centroid"], dtype="<f4", count=dimension).copy()
        return SpeakerProfile(
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            speaker_id=row["speaker_id"],
            centroid=centroid,
            sample_count=int(row["sample_count"]),
            quality_ema=float(row["quality_ema"]),
            last_active_ms=int(row["last_active_ms"]),
            display_name=row["display_name"],
        )

