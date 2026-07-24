"""Rebuild the lightweight correction index after the LigEdit GUI exits."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time

from ligweb.inference import classify_batch_detailed, get_base_model_hash
from ligweb.correction_model import (
    CorrectionIndex,
    CorrectionRow,
    activate_generation,
    build_candidate,
    is_candidate_acceptable,
    load_active_index,
    save_generation,
)
from ligweb.feedback_store import FeedbackStore, default_feedback_dir


@dataclass(frozen=True)
class TrainingOutcome:
    status: str
    generation: int
    record_count: int
    reason: str


def _timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _configure_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ligweb.feedback_training")
    if not logger.handlers:
        handler = logging.FileHandler(root / "training.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def training_lock(root: Path):
    path = root / "training.lock"
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if attempt or time.time() - path.stat().st_mtime < 7200:
                raise
            path.unlink(missing_ok=True)
    else:
        raise FileExistsError(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()} {_timestamp()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def _next_generation(root: Path, store: FeedbackStore) -> int:
    generations = [int(store.get_state("active_generation") or 0)]
    models = root / "models"
    if models.exists():
        for path in models.glob("generation-*"):
            try:
                generations.append(int(path.name.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(generations) + 1


def _write_status(store, status, reason):
    store.set_state("last_training_status", status)
    store.set_state("last_training_reason", reason)
    store.set_state("last_training_time", _timestamp())


def run_feedback_training(feedback_dir=None) -> TrainingOutcome:
    root = Path(feedback_dir) if feedback_dir else default_feedback_dir()
    store = FeedbackStore(root / "feedback.sqlite3")
    generation = int(store.get_state("active_generation") or 0)
    if not store.is_dirty():
        enabled_count = store.count_records(enabled_only=True)
        current_hash = get_base_model_hash()
        compatible = (
            enabled_count == 0
            or load_active_index(root, current_hash) is not None
        )
        if compatible:
            reason = "没有新增或修改的纠错记录"
            _write_status(store, "no_changes", reason)
            return TrainingOutcome(
                "no_changes", generation, enabled_count, reason
            )

    logger = _configure_logging(root)
    try:
        with training_lock(root):
            records, failures = store.list_records_with_failures(enabled_only=True)
            for waveform_hash, error in failures:
                logger.warning("skip corrupt feedback %s: %s", waveform_hash, error)

            model_hash = get_base_model_hash()
            if not model_hash:
                raise RuntimeError("base model is unavailable")
            next_generation = _next_generation(root, store)
            active = load_active_index(root, model_hash)

            if records:
                predictions = classify_batch_detailed(
                    [record.waveform for record in records],
                    daylights=[
                        _event_is_daylight(record.event_time) for record in records
                    ],
                )
                rows = [
                    CorrectionRow(
                        prediction.feature,
                        prediction.label,
                        record.corrected_label,
                    )
                    for prediction, record in zip(predictions, records)
                ]
                candidate = build_candidate(rows, model_hash, next_generation)
                acceptable = is_candidate_acceptable(candidate, active, rows)
            else:
                rows = []
                candidate = CorrectionIndex.empty(model_hash, next_generation)
                acceptable = is_candidate_acceptable(
                    candidate, active, rows, all_records_cancelled=True
                )

            if acceptable:
                directory = save_generation(root, candidate)
                activate_generation(root, directory)
                generation = candidate.generation
                status = "activated"
                reason = (
                    f"records={len(rows)} threshold={candidate.threshold} "
                    f"precision={candidate.validation_precision:.3f}"
                )
            else:
                status = "retained"
                reason = "candidate did not improve the active correction model"
            if status == "activated":
                store.mark_clean(generation)
            else:
                store.mark_reviewed()
            _write_status(store, status, reason)
            logger.info("feedback training %s: %s", status, reason)
            return TrainingOutcome(status, generation, len(rows), reason)
    except FileExistsError:
        reason = "another feedback trainer is running"
        _write_status(store, "locked", reason)
        return TrainingOutcome("locked", generation, 0, reason)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        logger.exception("feedback training failed")
        _write_status(store, "failed", reason)
        return TrainingOutcome("failed", generation, 0, reason)


def _event_is_daylight(event_time: str) -> bool:
    """Match the main model's UTC-to-China daylight feature."""
    try:
        hour = int(str(event_time)[6:8])
        minute = int(str(event_time)[8:10])
    except (TypeError, ValueError):
        return False
    local_hour = (hour + 8 + minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0
