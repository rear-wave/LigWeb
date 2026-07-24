"""Promote reviewed IC corrections into the primary training dataset."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock

from ligweb.correction_dataset import (
    default_lig_name,
    read_stored_lig,
    write_stored_lig,
)


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
        yield from sorted(path for path in root.rglob("*.lig") if path.is_file())


def _remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


class ICCorrectionPromoter:
    """Move correction IC pieces into train_data/IC without waveform duplicates."""

    def __init__(
        self,
        correction_data_dir: Path,
        train_data_dir: Path,
        status_path: Path | None = None,
    ) -> None:
        self.source_root = Path(correction_data_dir).resolve() / "IC"
        self.target_root = Path(train_data_dir).resolve() / "IC"
        self.status_path = (
            Path(status_path).resolve() if status_path is not None else None
        )
        self._lock = Lock()

    def status(self) -> dict:
        if self.status_path is None or not self.status_path.is_file():
            return {
                "status": "waiting",
                "reason": "等待每天 22:00 主模型训练前迁移 IC",
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {
                "status": "failed",
                "reason": "IC 迁移状态文件无法读取",
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }

    def audit(self) -> dict:
        """Inspect the pending migration without changing either dataset."""
        return self._run(apply=False)

    def promote(self) -> dict:
        """Move pending IC data, deleting each source only after validation."""
        if not self._lock.acquire(blocking=False):
            return {**self.status(), "running": True}
        try:
            result = self._run(apply=True)
        except Exception as error:
            result = {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "time": _timestamp(),
                "running": False,
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }
        finally:
            self._lock.release()
        if self.status_path is not None:
            _atomic_json(self.status_path, result)
        return result

    def _run(self, *, apply: bool) -> dict:
        if apply:
            self.target_root.mkdir(parents=True, exist_ok=True)
        training_hashes: set[str] = set()
        for path in _files(self.target_root):
            training_hashes.update(
                piece.digest for piece in read_stored_lig(path).pieces
            )

        source_files = list(_files(self.source_root))
        source_pieces = 0
        moved_pieces = 0
        duplicate_pieces = 0
        deleted_files = 0
        target_files: set[str] = set()
        seen = set(training_hashes)

        for source in source_files:
            source_stat = source.stat()
            stored = read_stored_lig(source)
            source_pieces += len(stored.pieces)
            selected = []
            for piece in stored.pieces:
                if piece.digest in seen:
                    duplicate_pieces += 1
                    continue
                seen.add(piece.digest)
                selected.append(piece)

            if not apply:
                moved_pieces += len(selected)
                continue

            if selected:
                target = (self.target_root / default_lig_name(source.name)).resolve()
                target.relative_to(self.target_root)
                if target.is_file():
                    existing = read_stored_lig(target)
                    target_header = existing.header
                    combined = list(existing.pieces) + selected
                else:
                    target_header = stored.header
                    combined = selected
                write_stored_lig(target, target_header, combined)
                written_hashes = {
                    piece.digest for piece in read_stored_lig(target).pieces
                }
                expected_hashes = {piece.digest for piece in selected}
                if not expected_hashes.issubset(written_hashes):
                    raise ValueError(f"迁移后的 LIG 校验失败: {target}")
                moved_pieces += len(selected)
                target_files.add(target.name)

            current_stat = source.stat()
            if (
                current_stat.st_size != source_stat.st_size
                or current_stat.st_mtime_ns != source_stat.st_mtime_ns
            ):
                raise RuntimeError(
                    f"迁移期间纠错文件发生变化，已保留源文件: {source}"
                )
            source.unlink()
            deleted_files += 1

        if apply:
            _remove_empty_directories(self.source_root)

        status = "promoted" if apply else "audited"
        if not source_files:
            reason = "纠错集中没有待迁移的 IC 数据"
        elif apply:
            reason = "IC 已去重迁移到训练集，纠错集源文件已删除"
        else:
            reason = "IC 迁移检查完成，未修改数据"
        return {
            "status": status,
            "reason": reason,
            "time": _timestamp(),
            "running": False,
            "source_files": len(source_files),
            "source_pieces": source_pieces,
            "moved_pieces": moved_pieces,
            "duplicate_pieces": duplicate_pieces,
            "deleted_files": deleted_files,
            "target_files": sorted(target_files),
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
        }
