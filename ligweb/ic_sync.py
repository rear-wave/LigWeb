"""Mirror the approved IC correction dataset into the training dataset."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
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


def _signature(root: Path) -> tuple:
    rows = []
    for file_path in _files(root):
        stat = file_path.stat()
        rows.append(
            (
                file_path.relative_to(root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return tuple(rows)


def _dataset_counts(root: Path) -> tuple[int, int, int]:
    files = list(_files(root))
    pieces = 0
    invalid_files = 0
    for file_path in files:
        try:
            pieces += len(read_stored_lig(file_path).pieces)
        except Exception:
            invalid_files += 1
    return len(files), pieces, invalid_files


class ICDatasetMirror:
    """Safely replace train_data/IC with a deduplicated correction-data copy."""

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
                "reason": "等待每天 22:00 将纠错集 IC 复制到训练集",
                "source_retained": True,
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {
                "status": "failed",
                "reason": "IC 复制状态文件无法读取",
                "source_retained": True,
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }

    def audit(self) -> dict:
        """Inspect the next mirror operation without changing either dataset."""
        return self._run(apply=False)

    def synchronize(self) -> dict:
        """Replace training IC only after a complete staged copy validates."""
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
                "source_retained": True,
                "source_root": str(self.source_root),
                "target_root": str(self.target_root),
            }
        finally:
            self._lock.release()
        if self.status_path is not None:
            _atomic_json(self.status_path, result)
        return result

    def _snapshot(self) -> tuple[dict[Path, dict], dict]:
        source_files = list(_files(self.source_root))
        groups: dict[Path, dict] = {}
        seen: set[str] = set()
        source_pieces = 0
        duplicate_pieces = 0

        for source in source_files:
            stored = read_stored_lig(source)
            relative = source.relative_to(self.source_root)
            target_relative = relative.with_name(default_lig_name(source.name))
            group = groups.setdefault(
                target_relative,
                {"header": stored.header, "pieces": []},
            )
            for piece in stored.pieces:
                source_pieces += 1
                if piece.digest in seen:
                    duplicate_pieces += 1
                    continue
                seen.add(piece.digest)
                group["pieces"].append(piece)

        groups = {
            relative: group for relative, group in groups.items() if group["pieces"]
        }
        return groups, {
            "source_files": len(source_files),
            "source_pieces": source_pieces,
            "copied_files": len(groups),
            "copied_pieces": len(seen),
            "duplicate_pieces": duplicate_pieces,
        }

    def _write_staging(self, staging_root: Path, groups: dict[Path, dict]) -> None:
        expected_hashes: set[str] = set()
        for relative, group in groups.items():
            target = (staging_root / relative).resolve()
            target.relative_to(staging_root.resolve())
            write_stored_lig(target, group["header"], group["pieces"])
            expected_hashes.update(piece.digest for piece in group["pieces"])

        actual_hashes: set[str] = set()
        for target in _files(staging_root):
            actual_hashes.update(
                piece.digest for piece in read_stored_lig(target).pieces
            )
        if actual_hashes != expected_hashes:
            raise ValueError("复制后的 IC 训练集校验失败")

    def _replace_target(self, staging_root: Path) -> None:
        parent = self.target_root.parent
        previous_root = parent / ".ligweb-ic-previous"
        if previous_root.exists():
            if self.target_root.exists():
                shutil.rmtree(previous_root)
            else:
                os.replace(previous_root, self.target_root)

        had_target = self.target_root.exists()
        if had_target:
            os.replace(self.target_root, previous_root)
        try:
            os.replace(staging_root, self.target_root)
        except Exception:
            if had_target and previous_root.exists() and not self.target_root.exists():
                os.replace(previous_root, self.target_root)
            raise
        if previous_root.exists():
            shutil.rmtree(previous_root)

    def _run(self, *, apply: bool) -> dict:
        source_signature = _signature(self.source_root)
        groups, source_summary = self._snapshot()
        cleared_files, cleared_pieces, cleared_invalid_files = _dataset_counts(
            self.target_root
        )

        if apply:
            parent = self.target_root.parent
            parent.mkdir(parents=True, exist_ok=True)
            staging_root = Path(
                tempfile.mkdtemp(prefix=".ligweb-ic-staging-", dir=parent)
            ).resolve()
            try:
                self._write_staging(staging_root, groups)
                if _signature(self.source_root) != source_signature:
                    raise RuntimeError(
                        "复制期间纠错集 IC 发生变化，已保留原训练集"
                    )
                self._replace_target(staging_root)
            finally:
                if staging_root.exists():
                    shutil.rmtree(staging_root)

        status = "synced" if apply else "audited"
        reason = (
            "训练集 IC 已由纠错集完整重建；纠错集源数据已保留"
            if apply
            else "IC 复制检查完成，未修改数据"
        )
        return {
            "status": status,
            "reason": reason,
            "time": _timestamp(),
            "running": False,
            "changed": bool(
                cleared_files
                or cleared_pieces
                or source_summary["copied_files"]
                or source_summary["copied_pieces"]
            ),
            **source_summary,
            "cleared_files": cleared_files,
            "cleared_pieces": cleared_pieces,
            "cleared_invalid_files": cleared_invalid_files,
            "source_retained": True,
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
        }
