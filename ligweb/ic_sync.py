"""Reconcile approved IC corrections into a managed training-data subtree."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock

from ligweb.correction_dataset import read_stored_lig, write_stored_lig


MANAGED_DIRECTORY_NAME = "_ligweb_sync"


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _files(root: Path):
    if root.is_dir():
        yield from sorted(
            path for path in root.rglob("*.lig") if path.is_file()
        )


def _signature(root: Path, excluded_root: Path | None = None) -> tuple:
    rows = []
    for path in _files(root):
        resolved = path.resolve()
        if excluded_root is not None:
            try:
                resolved.relative_to(excluded_root)
            except ValueError:
                pass
            else:
                continue
        stat = path.stat()
        rows.append(
            (
                path.relative_to(root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return tuple(rows)


class ICDataSynchronizer:
    """Keep `train_data/IC/_ligweb_sync` aligned without waveform duplicates."""

    def __init__(
        self,
        correction_data_dir: Path,
        train_data_dir: Path,
        status_path: Path,
    ) -> None:
        self.source_root = Path(correction_data_dir).resolve() / "IC"
        self.train_ic_root = Path(train_data_dir).resolve() / "IC"
        self.managed_root = self.train_ic_root / MANAGED_DIRECTORY_NAME
        self.status_path = Path(status_path).resolve()
        self._lock = Lock()
        self._last_signature: tuple | None = None

    def status(self) -> dict:
        if not self.status_path.is_file():
            return {
                "status": "waiting",
                "reason": "等待首次 IC 自动同步",
                "managed_root": str(self.managed_root),
            }
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {
                "status": "failed",
                "reason": "IC 同步状态文件无法读取",
                "managed_root": str(self.managed_root),
            }

    def _current_signature(self) -> tuple:
        managed = self.managed_root.resolve()
        return (
            _signature(self.source_root),
            _signature(self.train_ic_root, managed),
        )

    def sync(self, *, force: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            return {**self.status(), "running": True}
        try:
            signature = self._current_signature()
            if not force and signature == self._last_signature:
                return {**self.status(), "changed": False, "running": False}
            result = self._reconcile()
            self._last_signature = self._current_signature()
            _atomic_json(self.status_path, result)
            return result
        except Exception as error:
            result = {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "time": _timestamp(),
                "running": False,
                "managed_root": str(self.managed_root),
            }
            _atomic_json(self.status_path, result)
            return result
        finally:
            self._lock.release()

    def _reconcile(self) -> dict:
        self.train_ic_root.mkdir(parents=True, exist_ok=True)
        existing_hashes: set[str] = set()
        for path in _files(self.train_ic_root):
            try:
                path.resolve().relative_to(self.managed_root.resolve())
            except ValueError:
                existing_hashes.update(
                    piece.digest for piece in read_stored_lig(path).pieces
                )

        old_hashes: set[str] = set()
        for path in _files(self.managed_root):
            old_hashes.update(
                piece.digest for piece in read_stored_lig(path).pieces
            )

        desired: dict[Path, tuple[bytes, list]] = {}
        source_files = 0
        source_pieces = 0
        skipped_existing = 0
        seen = set(existing_hashes)
        for source in _files(self.source_root):
            stored = read_stored_lig(source)
            source_files += 1
            source_pieces += len(stored.pieces)
            selected = []
            for piece in stored.pieces:
                if piece.digest in seen:
                    skipped_existing += 1
                    continue
                seen.add(piece.digest)
                selected.append(piece)
            if selected:
                relative = source.relative_to(self.source_root)
                desired[relative] = (stored.header, selected)

        self.managed_root.mkdir(parents=True, exist_ok=True)
        desired_paths: set[Path] = set()
        synced_hashes: set[str] = set()
        for relative, (header, pieces) in desired.items():
            target = (self.managed_root / relative).resolve()
            target.relative_to(self.managed_root.resolve())
            write_stored_lig(target, header, pieces)
            desired_paths.add(target)
            synced_hashes.update(piece.digest for piece in pieces)

        removed_files = 0
        for path in list(_files(self.managed_root)):
            if path.resolve() not in desired_paths:
                path.unlink()
                removed_files += 1
        directories = sorted(
            (
                path
                for path in self.managed_root.rglob("*")
                if path.is_dir()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass

        changed = old_hashes != synced_hashes
        return {
            "status": "synced",
            "reason": "IC 纠错数据已与训练集同步",
            "time": _timestamp(),
            "running": False,
            "changed": changed,
            "source_files": source_files,
            "source_pieces": source_pieces,
            "synced_files": len(desired),
            "synced_pieces": len(synced_hashes),
            "skipped_existing_pieces": skipped_existing,
            "removed_synced_pieces": len(old_hashes - synced_hashes),
            "removed_files": removed_files,
            "managed_root": str(self.managed_root),
        }
