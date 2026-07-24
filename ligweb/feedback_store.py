"""Persistent, local storage for waveform-label feedback."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import sqlite3
import zlib

import numpy as np


CLASS_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
CORRECTION_TRAINING_REVISION = "large-dataset-v3"


@dataclass(frozen=True)
class FeedbackRecord:
    waveform_hash: str
    waveform: np.ndarray
    source_name: str
    piece_index: int
    event_time: str
    base_model_hash: str
    base_label: str
    base_confidence: float
    probabilities: tuple[float, float, float, float, float]
    corrected_label: str
    enabled: bool
    created_at: str
    updated_at: str
    trained_generation: int | None


def default_feedback_dir() -> Path:
    configured = os.environ.get("LIGWEB_FEEDBACK_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "runtime"


def _canonical_waveform(waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(waveform)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional array")
    return np.ascontiguousarray(values, dtype="<u2")


def waveform_digest(waveform: np.ndarray) -> str:
    values = _canonical_waveform(waveform)
    payload = len(values).to_bytes(8, "little") + values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class FeedbackStore:
    """SQLite-backed feedback records, keyed by canonical waveform content."""

    def __init__(self, database_path: str | Path | None = None):
        if database_path is None:
            database_path = default_feedback_dir() / "feedback.sqlite3"
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_records (
                    waveform_hash TEXT PRIMARY KEY,
                    waveform_blob BLOB NOT NULL,
                    sample_count INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    piece_index INTEGER NOT NULL,
                    event_time TEXT NOT NULL,
                    base_model_hash TEXT NOT NULL,
                    base_label TEXT NOT NULL,
                    base_confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trained_generation INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO app_state(key, value) VALUES('dirty', '0')"
            )

    @staticmethod
    def _validate_label(label: str, field_name: str) -> None:
        if label not in CLASS_NAMES:
            raise ValueError(f"{field_name} must be one of {CLASS_NAMES}")

    @classmethod
    def _validate_probabilities(
        cls, probabilities: tuple[float, float, float, float, float]
    ) -> tuple[float, float, float, float, float]:
        values = tuple(float(value) for value in probabilities)
        if len(values) != len(CLASS_NAMES):
            raise ValueError("probabilities must contain five values")
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("probabilities must be finite values from zero to one")
        return values

    @staticmethod
    def _decode_waveform(blob: bytes, sample_count: int) -> np.ndarray:
        if type(sample_count) is not int or sample_count <= 0:
            raise ValueError("invalid compressed waveform")
        try:
            decoded = zlib.decompress(blob)
        except (TypeError, zlib.error) as error:
            raise ValueError("invalid compressed waveform") from error
        if len(decoded) != sample_count * 2:
            raise ValueError("invalid compressed waveform")
        return np.frombuffer(decoded, dtype="<u2").copy()

    @classmethod
    def _record_from_row(cls, row: sqlite3.Row) -> FeedbackRecord:
        waveform = cls._decode_waveform(row["waveform_blob"], row["sample_count"])
        try:
            probabilities = tuple(float(value) for value in json.loads(row["probabilities"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid probabilities") from error
        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError("invalid probabilities")
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("invalid probabilities")
        cls._validate_label(row["base_label"], "base_label")
        cls._validate_label(row["corrected_label"], "corrected_label")
        return FeedbackRecord(
            waveform_hash=row["waveform_hash"],
            waveform=waveform,
            source_name=row["source_name"],
            piece_index=row["piece_index"],
            event_time=row["event_time"],
            base_model_hash=row["base_model_hash"],
            base_label=row["base_label"],
            base_confidence=row["base_confidence"],
            probabilities=probabilities,
            corrected_label=row["corrected_label"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            trained_generation=row["trained_generation"],
        )

    def upsert_feedback(
        self, waveform: np.ndarray, source_name: str, piece_index: int,
        event_time: str, base_model_hash: str, base_label: str,
        base_confidence: float, probabilities: tuple[float, float, float, float, float],
        corrected_label: str,
    ) -> FeedbackRecord:
        values = _canonical_waveform(waveform)
        self._validate_label(base_label, "base_label")
        self._validate_label(corrected_label, "corrected_label")
        probability_values = self._validate_probabilities(probabilities)
        waveform_hash = waveform_digest(values)
        now = _utc_now()
        blob = zlib.compress(values.tobytes())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO feedback_records(
                    waveform_hash, waveform_blob, sample_count, source_name, piece_index,
                    event_time, base_model_hash, base_label, base_confidence, probabilities,
                    corrected_label, enabled, created_at, updated_at, trained_generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)
                ON CONFLICT(waveform_hash) DO UPDATE SET
                    waveform_blob=excluded.waveform_blob,
                    sample_count=excluded.sample_count,
                    source_name=excluded.source_name,
                    piece_index=excluded.piece_index,
                    event_time=excluded.event_time,
                    base_model_hash=excluded.base_model_hash,
                    base_label=excluded.base_label,
                    base_confidence=excluded.base_confidence,
                    probabilities=excluded.probabilities,
                    corrected_label=excluded.corrected_label,
                    enabled=1,
                    updated_at=excluded.updated_at,
                    trained_generation=NULL
                """,
                (waveform_hash, blob, len(values), Path(source_name).name,
                 piece_index, event_time,
                 base_model_hash, base_label, float(base_confidence),
                 json.dumps(probability_values), corrected_label, now, now),
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('dirty', '1')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_status', 'pending')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_reason', '等待纠错模型训练')"
            )
            row = connection.execute(
                "SELECT * FROM feedback_records WHERE waveform_hash=?", (waveform_hash,)
            ).fetchone()
        return self._record_from_row(row)

    def get_record(self, waveform_hash: str) -> FeedbackRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_records WHERE waveform_hash=?", (waveform_hash,)
            ).fetchone()
        return None if row is None else self._record_from_row(row)

    def list_records(self, enabled_only: bool = False) -> list[FeedbackRecord]:
        records, _ = self.list_records_with_failures(enabled_only)
        return records

    def count_records(self, enabled_only: bool = False) -> int:
        query = "SELECT COUNT(*) FROM feedback_records"
        if enabled_only:
            query += " WHERE enabled=1"
        with self.connect() as connection:
            return int(connection.execute(query).fetchone()[0])

    def list_records_with_failures(
        self, enabled_only: bool = False
    ) -> tuple[list[FeedbackRecord], list[tuple[str, str]]]:
        query = "SELECT * FROM feedback_records"
        if enabled_only:
            query += " WHERE enabled=1"
        query += " ORDER BY created_at, waveform_hash"
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        records = []
        failures = []
        for row in rows:
            try:
                records.append(self._record_from_row(row))
            except ValueError as error:
                failures.append((row["waveform_hash"], str(error)))
        return records, failures

    def cancel_feedback(self, waveform_hash: str) -> bool:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                "UPDATE feedback_records SET enabled=0, updated_at=? WHERE waveform_hash=?",
                (_utc_now(), waveform_hash),
            )
            if not result.rowcount:
                return False
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('dirty', '1')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_status', 'pending')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_reason', '等待纠错模型训练')"
            )
        return True

    def reenable_feedback(
        self, waveform_hash: str, corrected_label: str
    ) -> FeedbackRecord:
        self._validate_label(corrected_label, "corrected_label")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                """UPDATE feedback_records
                   SET corrected_label=?, enabled=1, updated_at=?, trained_generation=NULL
                   WHERE waveform_hash=?""",
                (corrected_label, _utc_now(), waveform_hash),
            )
            if not result.rowcount:
                raise KeyError(waveform_hash)
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('dirty', '1')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_status', 'pending')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('last_training_reason', '等待纠错模型训练')"
            )
            row = connection.execute(
                "SELECT * FROM feedback_records WHERE waveform_hash=?", (waveform_hash,)
            ).fetchone()
        return self._record_from_row(row)

    def get_state(self, key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_state WHERE key=?", (key,)
            ).fetchone()
        return None if row is None else row["value"]

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES(?, ?)",
                (str(key), str(value)),
            )

    def is_dirty(self) -> bool:
        if self.get_state("dirty") == "1":
            return True
        if self.get_state("training_revision") == CORRECTION_TRAINING_REVISION:
            return False
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM feedback_records WHERE enabled=1 LIMIT 1"
            ).fetchone()
        return row is not None

    def mark_clean(self, generation: int) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE feedback_records SET trained_generation=? WHERE enabled=1",
                (generation,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('active_generation', ?)",
                (str(generation),),
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('dirty', '0')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('training_revision', ?)",
                (CORRECTION_TRAINING_REVISION,),
            )

    def mark_reviewed(self) -> None:
        """Clear the dirty flag without claiming records entered an adapter."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) VALUES('dirty', '0')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO app_state(key, value) "
                "VALUES('training_revision', ?)",
                (CORRECTION_TRAINING_REVISION,),
            )

    def export_csv(self, output_path: str | Path) -> None:
        field_names = [
            "waveform_hash", "source_name", "piece_index", "event_time",
            "base_model_hash", "base_label", "base_confidence", "probabilities",
            "corrected_label", "enabled", "created_at", "updated_at",
            "trained_generation",
        ]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for record in self.list_records():
                writer.writerow({
                    "waveform_hash": record.waveform_hash,
                    "source_name": Path(record.source_name).name,
                    "piece_index": record.piece_index,
                    "event_time": record.event_time,
                    "base_model_hash": record.base_model_hash,
                    "base_label": record.base_label,
                    "base_confidence": record.base_confidence,
                    "probabilities": json.dumps(record.probabilities),
                    "corrected_label": record.corrected_label,
                    "enabled": int(record.enabled),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "trained_generation": record.trained_generation,
                })
