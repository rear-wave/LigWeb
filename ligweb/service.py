"""Synchronous LigWeb application service shared by the API and tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePath
import re
import tempfile
from threading import Lock, RLock, Thread
from typing import Any
import zipfile

import numpy as np

from ligweb.inference import (
    apply_feedback_batch,
    classify_batch_detailed,
    get_base_model_hash,
    load_correction_context,
)
from ligweb.correction_dataset import (
    StoredPiece,
    read_stored_lig,
    remove_waveforms,
    waveform_index,
    write_stored_lig,
)
from ligweb.feedback_store import CLASS_NAMES, FeedbackStore, waveform_digest
from ligweb.feedback_training import run_feedback_training
from ligweb.ic_sync import ICDataSynchronizer
from ligweb.lig_io import save_lig_file
from ligweb.lig_parser import (
    ButterFilter,
    ReadLigFileWithOffsets,
    format_time_display,
    time_classifier_display,
)
from ligweb.config import LigWebConfig
from ligweb.scheduler import (
    CHINA_TIMEZONE,
    CorrectionTrainingScheduler,
    PeriodicTaskScheduler,
    next_daily_22,
    next_hour,
)


@dataclass
class LigDocument:
    path: Path
    mtime_ns: int
    header: dict
    pieces: list
    raw_data: bytes
    piece_offsets: list
    header_size: int
    base_predictions: list | None = None
    base_model_hash: str | None = None


def _json_scalar(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return value


def _safe_output_name(name: str) -> str:
    name = PurePath(name).name
    if not name.lower().endswith(".lig"):
        name += ".lig"
    name = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]", "_", name)
    if name in {"", ".lig"}:
        raise ValueError("invalid output filename")
    return name


def _safe_archive_name(name: str) -> str:
    name = PurePath(name).name
    if not name.lower().endswith(".zip"):
        name += ".zip"
    name = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]", "_", name)
    if name in {"", ".zip"}:
        raise ValueError("invalid archive filename")
    return name


def _decimate(values: np.ndarray, max_points: int) -> tuple[list[int], list[float]]:
    """Min/max envelope decimation that keeps narrow lightning peaks visible."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= max_points:
        return list(range(len(values))), values.tolist()
    bucket_count = max(1, max_points // 2)
    edges = np.linspace(0, len(values), bucket_count + 1, dtype=np.int64)
    positions: list[int] = []
    samples: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        bucket = values[start:end]
        minimum = int(np.argmin(bucket)) + int(start)
        maximum = int(np.argmax(bucket)) + int(start)
        for position in sorted((minimum, maximum)):
            positions.append(position)
            samples.append(float(values[position]))
    return positions, samples


class LigWebService:
    def __init__(self, config: LigWebConfig):
        self.config = config
        self.config.ensure_directories()
        self.feedback_store = FeedbackStore(
            self.config.feedback_dir / "feedback.sqlite3"
        )
        self._cache: OrderedDict[Path, LigDocument] = OrderedDict()
        self._cache_lock = RLock()
        self._document_write_lock = Lock()
        self._correction_write_lock = Lock()
        self._inference_lock = Lock()
        self._correction_context = None
        self._correction_signature = None
        self._inference_revision = 0
        self._training_lock = Lock()
        self._training_thread: Thread | None = None
        self._upload_lock = Lock()
        os.environ.setdefault(
            "LIGWEB_BASE_MODEL_PATH", str(self.config.main_model_path)
        )
        os.environ.setdefault(
            "LIGWEB_BASE_MODEL_METADATA_PATH",
            str(self.config.main_model_metadata_path),
        )
        os.environ.setdefault(
            "LIGWEB_CORRECTION_MODEL_DIR",
            str(self.config.correction_model_dir),
        )
        self._correction_scheduler = CorrectionTrainingScheduler(
            self.start_training,
            lambda: self.feedback_store.get_state("auto_correction_slot"),
            lambda value: self.feedback_store.set_state(
                "auto_correction_slot", value
            ),
            enabled=self.config.auto_correction_training,
        )
        self._ic_synchronizer = ICDataSynchronizer(
            self.config.correction_data_dir,
            self.config.train_data_dir,
            self.config.ic_sync_status_path,
        )
        self._ic_sync_scheduler = PeriodicTaskScheduler(
            self.sync_ic_data,
            enabled=self.config.auto_ic_sync,
            poll_seconds=self.config.ic_sync_poll_seconds,
            thread_name="ligweb-ic-sync",
        )

    def start_scheduler(self) -> None:
        self._correction_scheduler.start()
        self._ic_sync_scheduler.start()

    def stop_scheduler(self) -> None:
        self._correction_scheduler.stop()
        self._ic_sync_scheduler.stop()

    def sync_ic_data(self, force: bool = False) -> dict:
        """Synchronize approved IC corrections into the managed train subtree."""
        if not self.config.auto_ic_sync and not force:
            return {
                **self._ic_synchronizer.status(),
                "status": "disabled",
                "reason": "IC 自动同步已关闭",
            }
        return self._ic_synchronizer.sync(force=force)

    def ic_sync_status(self) -> dict:
        return self._ic_synchronizer.status()

    def _resolve_dataset_file(self, dataset: str, relative_path: str) -> Path:
        root = self.config.dataset_root(dataset).resolve()
        candidate = (root / relative_path.replace("\\", "/")).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise PermissionError("path escapes configured dataset") from error
        if candidate.suffix.lower() != ".lig":
            raise ValueError("only .lig files are available")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def list_files(
        self,
        dataset: str,
        query: str = "",
        offset: int = 0,
        limit: int = 200,
    ) -> dict:
        root = self.config.dataset_root(dataset).resolve()
        query = query.strip().lower()
        offset = max(0, int(offset))
        limit = min(500, max(1, int(limit)))
        matches = []
        for path in root.rglob("*.lig"):
            relative = path.relative_to(root).as_posix()
            if relative.startswith(".ligedit/"):
                continue
            if dataset == "correction" and relative.startswith(
                ("uploads/", "exports/")
            ):
                continue
            if query and query not in relative.lower():
                continue
            matches.append((relative, path))
        matches.sort(key=lambda item: item[0].lower())
        page = matches[offset:offset + limit]
        return {
            "dataset": dataset,
            "total": len(matches),
            "offset": offset,
            "limit": limit,
            "files": [
                {
                    "path": relative,
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
                for relative, path in page
            ],
        }

    def _load_document(self, dataset: str, relative_path: str) -> LigDocument:
        path = self._resolve_dataset_file(dataset, relative_path)
        mtime_ns = path.stat().st_mtime_ns
        with self._cache_lock:
            cached = self._cache.get(path)
            if cached is not None and cached.mtime_ns == mtime_ns:
                self._cache.move_to_end(path)
                return cached

        header, pieces, raw_data, offsets, header_size = ReadLigFileWithOffsets(path)
        document = LigDocument(
            path=path,
            mtime_ns=mtime_ns,
            header=header,
            pieces=pieces,
            raw_data=raw_data,
            piece_offsets=offsets,
            header_size=header_size,
        )
        with self._cache_lock:
            self._cache[path] = document
            self._cache.move_to_end(path)
            while len(self._cache) > self.config.max_cached_files:
                self._cache.popitem(last=False)
        return document

    def close_document(self, dataset: str, relative_path: str) -> bool:
        """Release a parsed LIG document and its cached model predictions."""
        path = self._resolve_dataset_file(dataset, relative_path).resolve()
        with self._cache_lock:
            return self._cache.pop(path, None) is not None

    def save_document(
        self,
        dataset: str,
        relative_path: str,
        deleted_indices: list[int],
    ) -> dict:
        """Atomically remove marked pieces from a source LIG file."""
        document = self._load_document(dataset, relative_path)
        path = document.path.resolve()
        total = len(document.piece_offsets)
        deleted = sorted({int(index) for index in deleted_indices})
        if deleted and (deleted[0] < 0 or deleted[-1] >= total):
            raise ValueError("deleted_indices must select valid pieces")
        if not deleted:
            return {
                "path": relative_path,
                "deleted_count": 0,
                "piece_count": total,
                "size": path.stat().st_size,
                "backup_path": None,
            }

        expected_count = total - len(deleted)
        temporary_path = None
        with self._document_write_lock:
            if path.stat().st_mtime_ns != document.mtime_ns:
                with self._cache_lock:
                    self._cache.pop(path, None)
                raise ValueError("LIG file changed on disk; reload it before saving")
            output_fd, output_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".saving", dir=path.parent
            )
            os.close(output_fd)
            temporary_path = Path(output_name)
            try:
                save_lig_file(
                    temporary_path,
                    document.raw_data,
                    document.header_size,
                    document.piece_offsets,
                    deleted,
                )
                _header, pieces, _raw, _offsets, _header_size = (
                    ReadLigFileWithOffsets(temporary_path)
                )
                if len(pieces) != expected_count:
                    raise ValueError("saved LIG validation failed")
                if path.stat().st_mtime_ns != document.mtime_ns:
                    raise ValueError("LIG file changed on disk; reload it before saving")
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        with self._cache_lock:
            self._cache.pop(path, None)
        return {
            "path": relative_path,
            "deleted_count": len(deleted),
            "piece_count": expected_count,
            "size": path.stat().st_size,
            "backup_path": None,
        }

    @staticmethod
    def _waveforms(document: LigDocument) -> list[np.ndarray]:
        values = []
        for _time_key, piece in document.pieces:
            waveform = np.asarray(piece.get("0", []), dtype=np.uint16)
            values.append(waveform)
        return values

    def _base_predictions(self, document: LigDocument):
        active_hash = get_base_model_hash()
        if (
            document.base_predictions is not None
            and document.base_model_hash is None
        ):
            # Supports pre-classified/injected documents while normal parsed
            # documents always begin without predictions.
            document.base_model_hash = active_hash
        if (
            document.base_predictions is None
            or document.base_model_hash != active_hash
        ):
            with self._inference_lock:
                active_hash = get_base_model_hash()
                if (
                    document.base_predictions is None
                    or document.base_model_hash != active_hash
                ):
                    document.base_predictions = classify_batch_detailed(
                        self._waveforms(document),
                        batch_size=256,
                        daylights=[
                            time_classifier_display(str(event_time)) == "白天"
                            for event_time, _piece in document.pieces
                        ],
                    )
                    document.base_model_hash = get_base_model_hash()
        return document.base_predictions

    def _get_correction_context(self):
        database = self.config.feedback_dir / "feedback.sqlite3"
        active = self.config.correction_model_dir / "active.json"
        signature = (
            database.stat().st_mtime_ns if database.exists() else None,
            active.stat().st_mtime_ns if active.exists() else None,
            get_base_model_hash(),
        )
        with self._cache_lock:
            if (
                self._correction_context is None
                or signature != self._correction_signature
            ):
                self._correction_context = load_correction_context(
                    self.config.feedback_dir,
                    self.config.correction_model_dir,
                )
                self._correction_signature = signature
            return self._correction_context

    def _invalidate_correction_context(self):
        with self._cache_lock:
            self._correction_context = None
            self._correction_signature = None
            self._inference_revision += 1

    def _effective_predictions(self, document: LigDocument):
        waveforms = self._waveforms(document)
        context = self._get_correction_context()
        return apply_feedback_batch(
            waveforms,
            self._base_predictions(document),
            context=context,
        )

    @staticmethod
    def _correction_dataset_label(
        dataset: str, relative_path: str
    ) -> str | None:
        if dataset != "correction":
            return None
        parts = PurePath(relative_path.replace("\\", "/")).parts
        return parts[0] if parts and parts[0] in CLASS_NAMES else None

    @staticmethod
    def _apply_dataset_label(prediction, dataset_label: str | None):
        """Treat a correction-folder label as reviewed truth below manual feedback."""
        if dataset_label is None or prediction.source == "manual_exact":
            return prediction
        return replace(
            prediction,
            effective_label=dataset_label,
            source="dataset_label",
            correction_similarity=None,
        )

    @staticmethod
    def _piece_metadata(piece: dict) -> dict:
        fields = (
            "version",
            "m_samplingRate",
            "m_numOfData",
            "m_numOfChannel",
            "m_stationID",
            "m_stationName",
            "m_GPSCurrentLocationLat",
            "m_GPSCurrentLocationLon",
            "m_preTriggerNum",
            "m_Range",
        )
        return {field: _json_scalar(piece.get(field)) for field in fields}

    def list_pieces(
        self, dataset: str, relative_path: str, classify: bool = True
    ) -> dict:
        document = self._load_document(dataset, relative_path)
        predictions = self._effective_predictions(document) if classify else None
        dataset_label = self._correction_dataset_label(dataset, relative_path)
        summaries = []
        for index, (event_time, piece) in enumerate(document.pieces):
            item = {
                "index": index,
                "event_time": str(event_time),
                "display_time": format_time_display(str(event_time)),
                "daynight": time_classifier_display(str(event_time)),
                "sample_count": int(len(piece.get("0", ()))),
                "station_id": _json_scalar(piece.get("m_stationID")),
            }
            if predictions is not None and index < len(predictions):
                prediction = self._apply_dataset_label(
                    predictions[index], dataset_label
                )
                item["classification"] = {
                    "label": prediction.effective_label,
                    "base_label": prediction.base_label,
                    "confidence": prediction.base_confidence,
                    "source": prediction.source,
                    "similarity": prediction.correction_similarity,
                    "probabilities": dict(zip(CLASS_NAMES, prediction.probabilities)),
                    "main_model": {
                        "label": prediction.base_label,
                        "confidence": prediction.base_confidence,
                        "probabilities": dict(
                            zip(CLASS_NAMES, prediction.probabilities)
                        ),
                    },
                    "correction_model": {
                        "label": prediction.effective_label,
                        "applied": prediction.source != "base",
                        "source": prediction.source,
                        "similarity": prediction.correction_similarity,
                    },
                }
            summaries.append(item)
        return {
            "dataset": dataset,
            "path": relative_path,
            "header": {key: _json_scalar(value) for key, value in document.header.items()},
            "piece_count": len(summaries),
            "pieces": summaries,
        }

    def get_piece(
        self,
        dataset: str,
        relative_path: str,
        piece_index: int,
        max_points: int = 4000,
    ) -> dict:
        document = self._load_document(dataset, relative_path)
        if piece_index < 0 or piece_index >= len(document.pieces):
            raise IndexError("piece index is out of range")
        event_time, piece = document.pieces[piece_index]
        raw = np.asarray(piece.get("0", []), dtype=np.float64)
        if raw.size == 0:
            raise ValueError("piece has no waveform channel")
        centered = raw - float(np.mean(raw))
        try:
            filtered = np.asarray(ButterFilter(centered), dtype=np.float64)
        except Exception:
            filtered = centered
        max_points = min(12000, max(200, int(max_points)))
        positions, raw_values = _decimate(centered, max_points)
        filtered_values = [float(filtered[position]) for position in positions]
        sampling_rate = piece.get("m_samplingRate", 5_000_000)
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            sampling_rate = 5_000_000
        times = [position / float(sampling_rate) * 1000.0 for position in positions]

        base = self._base_predictions(document)[piece_index]
        effective = apply_feedback_batch(
            [raw],
            [base],
            context=self._get_correction_context(),
        )[0]
        effective = self._apply_dataset_label(
            effective,
            self._correction_dataset_label(dataset, relative_path),
        )
        return {
            "dataset": dataset,
            "path": relative_path,
            "index": piece_index,
            "event_time": str(event_time),
            "display_time": format_time_display(str(event_time)),
            "daynight": time_classifier_display(str(event_time)),
            "metadata": self._piece_metadata(piece),
            "waveform_hash": waveform_digest(raw),
            "waveform": {
                "time_ms": times,
                "raw": raw_values,
                "filtered": filtered_values,
                "original_sample_count": int(raw.size),
            },
            "classification": {
                "label": effective.effective_label,
                "base_label": effective.base_label,
                "confidence": effective.base_confidence,
                "source": effective.source,
                "similarity": effective.correction_similarity,
                "probabilities": dict(zip(CLASS_NAMES, effective.probabilities)),
                "main_model": {
                    "label": effective.base_label,
                    "confidence": effective.base_confidence,
                    "probabilities": dict(
                        zip(CLASS_NAMES, effective.probabilities)
                    ),
                },
                "correction_model": {
                    "label": effective.effective_label,
                    "applied": effective.source != "base",
                    "source": effective.source,
                    "similarity": effective.correction_similarity,
                    "generation": self._get_correction_context().generation,
                },
            },
        }

    def save_feedback(
        self,
        dataset: str,
        relative_path: str,
        piece_index: int,
        corrected_label: str,
    ) -> dict:
        if corrected_label not in CLASS_NAMES:
            raise ValueError(f"label must be one of {CLASS_NAMES}")
        document = self._load_document(dataset, relative_path)
        if piece_index < 0 or piece_index >= len(document.pieces):
            raise IndexError("piece index is out of range")
        event_time, piece = document.pieces[piece_index]
        waveform = np.asarray(piece.get("0", []), dtype=np.uint16)
        prediction = self._base_predictions(document)[piece_index]
        record = self.feedback_store.upsert_feedback(
            waveform=waveform,
            source_name=document.path.name,
            piece_index=piece_index,
            event_time=str(event_time),
            base_model_hash=get_base_model_hash(),
            base_label=prediction.label,
            base_confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            corrected_label=corrected_label,
        )
        self._invalidate_correction_context()
        return self._feedback_record_dict(record)

    def cancel_feedback(self, waveform_hash: str) -> bool:
        cancelled = self.feedback_store.cancel_feedback(waveform_hash)
        if cancelled:
            self._invalidate_correction_context()
        return cancelled

    @staticmethod
    def _feedback_record_dict(record) -> dict:
        return {
            "waveform_hash": record.waveform_hash,
            "source_name": record.source_name,
            "piece_index": record.piece_index,
            "event_time": record.event_time,
            "base_label": record.base_label,
            "base_confidence": record.base_confidence,
            "corrected_label": record.corrected_label,
            "enabled": record.enabled,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "trained_generation": record.trained_generation,
        }

    def list_feedback(self) -> list[dict]:
        return [
            self._feedback_record_dict(record)
            for record in reversed(self.feedback_store.list_records())
        ]

    def correction_training_status(self) -> dict:
        running = bool(
            self._training_thread is not None and self._training_thread.is_alive()
        )
        return {
            "running": running,
            "status": self.feedback_store.get_state("last_training_status")
            or "idle",
            "reason": self.feedback_store.get_state("last_training_reason") or "",
            "time": self.feedback_store.get_state("last_training_time"),
            "generation": int(
                self.feedback_store.get_state("active_generation") or 0
            ),
            "inference_revision": self._inference_revision,
            "dirty": self.feedback_store.is_dirty(),
            "record_count": self.feedback_store.count_records(),
        }

    def main_training_status(self) -> dict:
        status = {
            "status": "waiting",
            "running": False,
            "reason": "等待每日 22:00 自动训练",
            "time": None,
        }
        path = self.config.main_training_status_path
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    status.update(value)
            except (OSError, ValueError):
                status["status"] = "status_error"
                status["reason"] = "主模型状态文件无法读取"
        schedule_path = self.config.main_model_dir / "schedule.json"
        if schedule_path.is_file():
            try:
                schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
                updated_at = datetime.fromisoformat(schedule["updated_at"])
                recent = (
                    datetime.now(updated_at.tzinfo) - updated_at
                ).total_seconds() < 120
                status["scheduler_active"] = bool(
                    schedule.get("active") and recent
                )
                status["scheduler_mode"] = schedule.get("mode")
                status["scheduler_pid"] = schedule.get("pid")
            except (OSError, ValueError, AttributeError, KeyError, TypeError):
                status["scheduler_active"] = False
        status["model_hash"] = get_base_model_hash()
        status["model_available"] = self.config.main_model_path.is_file()
        return status

    def automation_status(self) -> dict:
        now = datetime.now(CHINA_TIMEZONE)
        return {
            "timezone": "Asia/Shanghai",
            "correction_enabled": self.config.auto_correction_training,
            "correction_schedule": "每逢整点",
            "next_correction_training": next_hour(now).isoformat(),
            "ic_sync_enabled": self.config.auto_ic_sync,
            "ic_sync_schedule": (
                f"每 {int(self.config.ic_sync_poll_seconds)} 秒检查"
            ),
            "main_schedule": "每天 22:00",
            "next_main_training": next_daily_22(now).isoformat(),
        }

    def training_status(self) -> dict:
        correction = self.correction_training_status()
        return {
            **correction,
            "correction": correction,
            "main": self.main_training_status(),
            "ic_sync": self._ic_synchronizer.status(),
            "automation": self.automation_status(),
        }

    def start_training(self, reason: str = "Web 端已提交训练") -> dict:
        with self._training_lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                return self.training_status()
            self.feedback_store.set_state("last_training_status", "queued")
            self.feedback_store.set_state("last_training_reason", reason)

            def train():
                with self._inference_lock:
                    run_feedback_training(
                        self.config.feedback_dir,
                        self.config.correction_model_dir,
                    )
                self._invalidate_correction_context()

            self._training_thread = Thread(
                target=train, name="ligweb-feedback-training", daemon=True
            )
            self._training_thread.start()
        return self.training_status()

    def import_corrected_pieces(
        self,
        dataset: str,
        relative_path: str,
        piece_indices: list[int],
    ) -> dict:
        """Write reviewed pieces into label-specific correction data."""
        document = self._load_document(dataset, relative_path)
        requested = sorted({int(index) for index in piece_indices})
        if not requested:
            raise ValueError("piece_indices must not be empty")
        if requested[0] < 0 or requested[-1] >= len(document.pieces):
            raise IndexError("piece index is out of range")

        predictions = self._effective_predictions(document)
        groups: dict[str, list[int]] = {}
        duplicate_skipped_indices: list[int] = []
        waveform_hashes: dict[int, str] = {}
        prepared = []
        manual_piece_count = 0
        model_piece_count = 0
        imported = []

        for index in requested:
            _event_time, piece = document.pieces[index]
            digest = waveform_digest(np.asarray(piece.get("0", [])))
            waveform_hashes[index] = digest
            prepared.append((index, digest, predictions[index]))

        with self._correction_write_lock:
            manual_hashes = {
                digest
                for _index, digest, prediction in prepared
                if prediction.source == "manual_exact"
            }
            reclassified_piece_count = (
                remove_waveforms(self.config.correction_data_dir, manual_hashes)
                if manual_hashes
                else 0
            )
            existing_hashes = waveform_index(self.config.correction_data_dir)
            for index, digest, prediction in prepared:
                if digest in existing_hashes:
                    duplicate_skipped_indices.append(index)
                    continue
                label = prediction.effective_label
                if label not in CLASS_NAMES:
                    raise ValueError(f"unsupported classification label: {label}")
                groups.setdefault(label, []).append(index)
                existing_hashes.add(digest)
                if prediction.source == "manual_exact":
                    manual_piece_count += 1
                else:
                    model_piece_count += 1

            for label, indices in sorted(groups.items()):
                output_dir = self.config.correction_data_dir / label
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = (output_dir / _safe_output_name(document.path.name)).resolve()
                output_path.relative_to(self.config.correction_data_dir.resolve())
                if output_path.exists():
                    existing = read_stored_lig(output_path)
                    header = existing.header
                    pieces = list(existing.pieces)
                else:
                    header = document.raw_data[:document.header_size]
                    pieces = []
                for index in indices:
                    event_time, _piece = document.pieces[index]
                    start, end = document.piece_offsets[index]
                    pieces.append(
                        StoredPiece(
                            str(event_time),
                            waveform_hashes[index],
                            document.raw_data[start:end],
                        )
                    )
                write_stored_lig(output_path, header, pieces)
                imported.append({
                    "label": label,
                    "path": output_path.relative_to(
                        self.config.correction_data_dir
                    ).as_posix(),
                    "piece_count": len(indices),
                    "size": output_path.stat().st_size,
                    "main_training_eligible": True,
                })

        ic_sync = None
        if self.config.auto_ic_sync and (imported or reclassified_piece_count):
            ic_sync = self.sync_ic_data(force=True)

        source_removed = False
        if dataset == "inbox":
            source_path = document.path.resolve()
            source_path.relative_to(self.config.inbox_dir.resolve())
            with self._cache_lock:
                self._cache.pop(source_path, None)
            source_path.unlink()
            source_removed = True
        return {
            "files": imported,
            "imported_piece_count": sum(
                item["piece_count"] for item in imported
            ),
            "manual_piece_count": manual_piece_count,
            "model_piece_count": model_piece_count,
            "duplicate_skipped_count": len(duplicate_skipped_indices),
            "duplicate_skipped_indices": duplicate_skipped_indices,
            "skipped_indices": duplicate_skipped_indices,
            "reclassified_piece_count": reclassified_piece_count,
            "ic_sync": ic_sync,
            "source_removed": source_removed,
        }

    def export_pieces(
        self,
        dataset: str,
        relative_path: str,
        keep_indices: list[int],
        output_name: str,
    ) -> dict:
        document = self._load_document(dataset, relative_path)
        total = len(document.piece_offsets)
        keep = {int(index) for index in keep_indices}
        if not keep or min(keep) < 0 or max(keep) >= total:
            raise ValueError("keep_indices must select valid pieces")
        deleted = sorted(set(range(total)) - keep)
        output_name = _safe_output_name(output_name)
        output_path = (self.config.exports_dir / output_name).resolve()
        output_path.relative_to(self.config.exports_dir.resolve())
        save_lig_file(
            output_path,
            document.raw_data,
            document.header_size,
            document.piece_offsets,
            deleted,
        )
        return {
            "name": output_name,
            "piece_count": len(keep),
            "size": output_path.stat().st_size,
        }

    def export_by_daynight(
        self,
        dataset: str,
        relative_path: str,
        excluded_indices: list[int],
        output_name: str,
    ) -> dict:
        """Export non-deleted pieces into day/night LIG files in one ZIP."""
        document = self._load_document(dataset, relative_path)
        total = len(document.piece_offsets)
        excluded = {int(index) for index in excluded_indices}
        if excluded and (min(excluded) < 0 or max(excluded) >= total):
            raise ValueError("excluded_indices must select valid pieces")
        groups = {"day": [], "night": []}
        for index, (event_time, _piece) in enumerate(document.pieces):
            if index in excluded:
                continue
            key = (
                "day"
                if time_classifier_display(str(event_time)) == "白天"
                else "night"
            )
            groups[key].append(index)
        if not any(groups.values()):
            raise ValueError("no pieces are available for export")

        output_name = _safe_archive_name(output_name)
        output_path = (self.config.exports_dir / output_name).resolve()
        output_path.relative_to(self.config.exports_dir.resolve())
        temporary_archive = output_path.with_name(f".{output_path.name}.tmp")
        archive_stem = output_path.stem
        if archive_stem.lower().endswith("_daynight"):
            archive_stem = archive_stem[:-9]
        try:
            with tempfile.TemporaryDirectory(
                prefix=".daynight-", dir=self.config.exports_dir
            ) as temporary_dir:
                temporary_root = Path(temporary_dir)
                with zipfile.ZipFile(
                    temporary_archive,
                    "w",
                    compression=zipfile.ZIP_STORED,
                ) as archive:
                    for key, keep_indices in groups.items():
                        if not keep_indices:
                            continue
                        lig_name = f"{archive_stem}_{key}.lig"
                        lig_path = temporary_root / lig_name
                        save_lig_file(
                            lig_path,
                            document.raw_data,
                            document.header_size,
                            document.piece_offsets,
                            sorted(set(range(total)) - set(keep_indices)),
                        )
                        archive.write(lig_path, arcname=lig_name)
            os.replace(temporary_archive, output_path)
        finally:
            temporary_archive.unlink(missing_ok=True)
        return {
            "name": output_name,
            "piece_count": len(groups["day"]) + len(groups["night"]),
            "day_count": len(groups["day"]),
            "night_count": len(groups["night"]),
            "size": output_path.stat().st_size,
        }

    def resolve_export(self, output_name: str) -> Path:
        output_name = (
            _safe_archive_name(output_name)
            if str(output_name).lower().endswith(".zip")
            else _safe_output_name(output_name)
        )
        path = (self.config.exports_dir / output_name).resolve()
        path.relative_to(self.config.exports_dir.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def upload_lig(self, filename: str, payload: bytes) -> dict:
        if not payload:
            raise ValueError("empty upload")
        filename = _safe_output_name(filename)
        upload_root = self.config.inbox_dir
        upload_root.mkdir(parents=True, exist_ok=True)
        with self._upload_lock:
            target = (upload_root / filename).resolve()
            target.relative_to(upload_root.resolve())
            if target.exists():
                stem = target.stem
                suffix = target.suffix
                sequence = 2
                while target.exists():
                    target = upload_root / f"{stem}_{sequence}{suffix}"
                    sequence += 1
            temporary = target.with_name(f".{target.name}.uploading")
            temporary.write_bytes(payload)
            try:
                ReadLigFileWithOffsets(temporary)
                os.replace(temporary, target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        return {
            "dataset": "inbox",
            "path": target.relative_to(upload_root).as_posix(),
            "name": target.name,
            "size": target.stat().st_size,
        }
