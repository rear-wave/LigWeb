"""Correction-dataset storage, deduplication, and migration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

from ligweb.feedback_store import CLASS_NAMES, waveform_digest
from ligweb.lig_parser import ReadLigFileWithOffsets


_GENERATED_SUFFIX_RE = re.compile(
    r"_(?:corrected|reviewed)_[0-9a-f]{12}$", re.IGNORECASE
)


@dataclass(frozen=True)
class StoredPiece:
    event_time: str
    digest: str
    data: bytes


@dataclass(frozen=True)
class StoredLig:
    path: Path
    header: bytes
    pieces: tuple[StoredPiece, ...]


def default_lig_name(name: str) -> str:
    """Restore generated import names to the original LIG filename style."""
    path = Path(name)
    stem = _GENERATED_SUFFIX_RE.sub("", path.stem)
    return f"{stem}.lig"


def iter_dataset_files(root: Path):
    """Yield only active correction data under the five class directories."""
    root = Path(root)
    for label in CLASS_NAMES:
        label_root = root / label
        if label_root.is_dir():
            yield from sorted(label_root.rglob("*.lig"))


def read_stored_lig(path: Path) -> StoredLig:
    header, pieces, raw, offsets, header_size = ReadLigFileWithOffsets(path)
    declared = int(header.get("NumOfPiece", -1))
    if declared != len(pieces) or len(pieces) != len(offsets):
        raise ValueError(
            f"cannot safely process {path}: declared {declared}, parsed {len(pieces)}"
        )
    stored = []
    for (event_time, piece), (start, end) in zip(pieces, offsets):
        digest = waveform_digest(np.asarray(piece.get("0", [])))
        stored.append(StoredPiece(str(event_time), digest, raw[start:end]))
    return StoredLig(Path(path), raw[:header_size], tuple(stored))


def waveform_index(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in iter_dataset_files(root):
        hashes.update(piece.digest for piece in read_stored_lig(path).pieces)
    return hashes


def write_stored_lig(path: Path, header: bytes, pieces: list[StoredPiece]) -> None:
    """Atomically write and validate a LIG assembled from raw piece records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray(header)
    struct.pack_into("<i", payload, 4, len(pieces))
    for piece in sorted(pieces, key=lambda item: item.event_time):
        payload.extend(piece.data)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        parsed = read_stored_lig(temporary)
        if len(parsed.pieces) != len(pieces):
            raise ValueError(f"failed to validate generated LIG: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_waveforms(root: Path, digests: set[str]) -> int:
    """Remove matching pieces from active correction files."""
    removed = 0
    for path in list(iter_dataset_files(root)):
        stored = read_stored_lig(path)
        kept = [piece for piece in stored.pieces if piece.digest not in digests]
        removed_here = len(stored.pieces) - len(kept)
        if not removed_here:
            continue
        removed += removed_here
        if kept:
            write_stored_lig(path, stored.header, kept)
        else:
            path.unlink()
    return removed


def _source_priority(path: Path, root: Path) -> tuple[int, str]:
    stem = path.stem.lower()
    if "_corrected_" in stem:
        priority = 0  # Old explicit manual output is the strongest label evidence.
    elif path.parent == root / path.relative_to(root).parts[0]:
        priority = 1
    elif "_reviewed_" in stem:
        priority = 2
    else:
        priority = 3
    return priority, path.as_posix().lower()


def deduplicate_dataset(
    root: Path,
    apply: bool = False,
) -> dict:
    """Flatten and globally deduplicate correction pieces by waveform content."""
    root = Path(root).resolve()
    if root.parent == root or not root.name:
        raise ValueError(f"unsafe correction dataset root: {root}")

    files = sorted(iter_dataset_files(root), key=lambda path: _source_priority(path, root))
    seen: dict[str, tuple[str, str]] = {}
    grouped: dict[tuple[str, str], dict] = {}
    duplicate_count = 0
    cross_label_duplicates = 0
    total_pieces = 0

    for path in files:
        relative = path.relative_to(root)
        label = relative.parts[0]
        stored = read_stored_lig(path)
        output_name = default_lig_name(path.name)
        key = label, output_name.lower()
        group = grouped.setdefault(
            key,
            {"label": label, "name": output_name, "header": stored.header, "pieces": []},
        )
        for piece in stored.pieces:
            total_pieces += 1
            previous = seen.get(piece.digest)
            if previous is not None:
                duplicate_count += 1
                if previous[0] != label:
                    cross_label_duplicates += 1
                continue
            seen[piece.digest] = (label, path.as_posix())
            group["pieces"].append(piece)

    result = {
        "root": str(root),
        "input_files": len(files),
        "input_pieces": total_pieces,
        "unique_pieces": len(seen),
        "duplicate_pieces_removed": duplicate_count,
        "cross_label_duplicates_removed": cross_label_duplicates,
        "output_files": sum(bool(group["pieces"]) for group in grouped.values()),
        "applied": False,
        "backup_dir": None,
    }
    if not apply:
        return result

    system_root = (root / ".ligedit").resolve()
    system_root.relative_to(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = system_root / "dedup-backups" / stamp
    staging_root = system_root / f"dedup-staging-{stamp}"
    if backup_root.exists() or staging_root.exists():
        raise FileExistsError(f"deduplication workspace already exists: {stamp}")

    staging_root.mkdir(parents=True)
    for label in CLASS_NAMES:
        (staging_root / label).mkdir()
    try:
        for group in grouped.values():
            if group["pieces"]:
                write_stored_lig(
                    staging_root / group["label"] / group["name"],
                    group["header"],
                    group["pieces"],
                )
        if len(waveform_index(staging_root)) != len(seen):
            raise ValueError("staged correction dataset failed deduplication validation")

        backup_root.mkdir(parents=True)
        moved_old: list[tuple[Path, Path]] = []
        installed: list[tuple[Path, Path]] = []
        try:
            for label in CLASS_NAMES:
                active = (root / label).resolve()
                active.relative_to(root)
                backup = backup_root / label
                staged = staging_root / label
                if active.exists():
                    shutil.move(str(active), str(backup))
                    moved_old.append((backup, active))
                shutil.move(str(staged), str(active))
                installed.append((active, staged))
        except Exception:
            for active, staged in reversed(installed):
                if active.exists():
                    shutil.move(str(active), str(staged))
            for backup, active in reversed(moved_old):
                if backup.exists():
                    shutil.move(str(backup), str(active))
            raise

        result["applied"] = True
        result["backup_dir"] = str(backup_root)
        (backup_root / "dedup-report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    finally:
        if staging_root.exists():
            staging_root.resolve().relative_to(system_root)
            shutil.rmtree(staging_root)


def restore_dataset_backup(root: Path, backup: Path) -> dict:
    """Restore a validated class-folder backup while snapshotting active data."""
    root = Path(root).resolve()
    backup = Path(backup).resolve()
    system_root = (root / ".ligedit").resolve()
    backups_root = (system_root / "dedup-backups").resolve()
    backup.relative_to(backups_root)
    if not backup.is_dir():
        raise FileNotFoundError(backup)

    audit = deduplicate_dataset(backup, apply=False)
    if audit["duplicate_pieces_removed"]:
        raise ValueError("refusing to restore a backup containing duplicate waveforms")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    active_backup = backups_root / f"{stamp}-before-restore"
    staging_root = system_root / f"restore-staging-{stamp}"
    if active_backup.exists() or staging_root.exists():
        raise FileExistsError(f"restore workspace already exists: {stamp}")
    staging_root.mkdir(parents=True)
    try:
        for label in CLASS_NAMES:
            source = backup / label
            staged = staging_root / label
            if source.is_dir():
                shutil.copytree(source, staged)
            else:
                staged.mkdir()
        staged_audit = deduplicate_dataset(staging_root, apply=False)
        if staged_audit["unique_pieces"] != audit["unique_pieces"]:
            raise ValueError("staged backup failed waveform-count validation")

        active_backup.mkdir(parents=True)
        moved_old: list[tuple[Path, Path]] = []
        installed: list[tuple[Path, Path]] = []
        try:
            for label in CLASS_NAMES:
                active = (root / label).resolve()
                active.relative_to(root)
                snapshot = active_backup / label
                staged = staging_root / label
                if active.exists():
                    shutil.move(str(active), str(snapshot))
                    moved_old.append((snapshot, active))
                shutil.move(str(staged), str(active))
                installed.append((active, staged))
        except Exception:
            for active, staged in reversed(installed):
                if active.exists():
                    shutil.move(str(active), str(staged))
            for snapshot, active in reversed(moved_old):
                if snapshot.exists():
                    shutil.move(str(snapshot), str(active))
            raise

        result = {
            "restored_from": str(backup),
            "active_backup": str(active_backup),
            "files": audit["input_files"],
            "pieces": audit["unique_pieces"],
        }
        (active_backup / "restore-report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    finally:
        if staging_root.exists():
            staging_root.resolve().relative_to(system_root)
            shutil.rmtree(staging_root)
