"""Nightly main-model trainer using train_data plus approved corrections."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import filecmp
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Callable

from ligweb.feedback_store import CLASS_NAMES
from ligweb.ic_sync import ICCorrectionPromoter
from tools.export_ligclassify_model import _work_around_broken_windows_asyncio


CHINA_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")
_LIG_FILE_HEADER_BYTES = 112
_LIG_PIECE_BYTES = 32208
_LIG_PIECE_COUNT_OFFSET = 4
_MAX_LIG_PIECES = 512


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


class MainTrainingStatus:
    def __init__(self, path: Path):
        self.path = path
        self.started_at = _timestamp()

    def write(self, status: str, reason: str, **extra) -> None:
        _atomic_json(
            self.path,
            {
                "status": status,
                "running": status in {"preparing", "training", "exporting"},
                "reason": reason,
                "time": _timestamp(),
                "started_at": self.started_at,
                **extra,
            },
        )


class MainTrainingLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            if time.time() - self.path.stat().st_mtime < 36 * 60 * 60:
                raise RuntimeError("另一个主模型训练任务仍在运行")
            self.path.unlink(missing_ok=True)
            self.descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        os.write(self.descriptor, f"{os.getpid()} {_timestamp()}\n".encode())
        os.close(self.descriptor)
        self.descriptor = None
        return self

    def __exit__(self, *_args):
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def _replace_dataset_view(view: Path) -> None:
    if view.exists():
        resolved = view.resolve()
        expected_parent = view.parent.resolve()
        if resolved.parent != expected_parent or view.name != "dataset":
            raise RuntimeError(f"拒绝清理意外的数据视图路径: {view}")
        shutil.rmtree(view)
    view.mkdir(parents=True)


def _validate_lig_file(path: Path) -> bool:
    """Validate a LIG and normalize a stale piece count when safe.

    Older LigEdit exports copied the source header without updating its piece
    count. When the payload is an exact number of complete pieces, correcting
    those four header bytes is lossless and makes the file valid for the strict
    ligClassify loader. The original modification time is preserved.
    """
    original_stat = path.stat()
    with path.open("rb") as handle:
        header = handle.read(_LIG_FILE_HEADER_BYTES)
    if len(header) != _LIG_FILE_HEADER_BYTES:
        raise ValueError("LIG 文件头不完整")
    declared_count = struct.unpack_from(
        "<i", header, _LIG_PIECE_COUNT_OFFSET
    )[0]
    payload_size = original_stat.st_size - _LIG_FILE_HEADER_BYTES
    actual_count, remainder = divmod(payload_size, _LIG_PIECE_BYTES)
    if payload_size < 0 or remainder or not 0 <= actual_count <= _MAX_LIG_PIECES:
        raise ValueError(
            f"LIG 数据区不是完整片段: 文件大小 {original_stat.st_size} 字节"
        )
    if declared_count == actual_count:
        return False
    with path.open("r+b") as handle:
        handle.seek(_LIG_PIECE_COUNT_OFFSET)
        handle.write(struct.pack("<i", actual_count))
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(
        path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    return True


def _link_labelled_root(
    source_root: Path,
    view: Path,
    group: str,
    validator: Callable[[Path], bool | None] | None = None,
    labels=CLASS_NAMES,
) -> tuple[int, list[dict[str, str]], list[str]]:
    count = 0
    invalid_files: list[dict[str, str]] = []
    repaired_headers: list[str] = []
    for label in labels:
        label_root = source_root / label
        if not label_root.is_dir():
            continue
        for source in label_root.rglob("*.lig"):
            relative = source.relative_to(label_root)
            if validator is not None:
                try:
                    repaired = validator(source)
                except (OSError, ValueError, struct.error) as error:
                    invalid_files.append(
                        {
                            "path": f"{label}/{relative.as_posix()}",
                            "reason": str(error),
                        }
                    )
                    continue
                if repaired:
                    repaired_headers.append(f"{label}/{relative.as_posix()}")
            target = view / label / group / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError as error:
                raise RuntimeError(
                    f"无法为训练数据创建硬链接（源和纠错目录需位于同一磁盘）: "
                    f"{source}"
                ) from error
            count += 1
    return count, invalid_files, repaired_headers


def _eligible_for_main_training(label: str, _relative: Path) -> bool:
    return label in CLASS_NAMES


def _unique_destination(target: Path, source: Path) -> tuple[Path, bool]:
    """Return a collision-safe destination and whether it is already copied."""
    if not target.exists():
        return target, False
    if filecmp.cmp(source, target, shallow=False):
        return target, True
    for index in range(2, 10_000):
        candidate = target.with_name(f"{target.stem}__{index}{target.suffix}")
        if not candidate.exists():
            return candidate, False
        if filecmp.cmp(source, candidate, shallow=False):
            return candidate, True
    raise RuntimeError(f"无法为纠错文件生成唯一名称: {target}")


def promote_daily_corrections(
    correction_root: Path,
    train_root: Path,
    local_date: date | None = None,
) -> dict:
    """Move today's explicitly imported corrections into the training set."""
    local_date = local_date or datetime.now(CHINA_TIMEZONE).date()
    moved: list[str] = []
    already_present: list[str] = []
    skipped_ineligible: list[str] = []
    failures: list[dict[str, str]] = []
    for label in CLASS_NAMES:
        if label == "IC":
            continue
        imports_root = correction_root / label / "imports"
        if not imports_root.is_dir():
            continue
        for source in sorted(imports_root.rglob("*.lig")):
            relative = source.relative_to(imports_root)
            try:
                modified_date = datetime.fromtimestamp(
                    source.stat().st_mtime, CHINA_TIMEZONE
                ).date()
                if modified_date != local_date:
                    continue
                display_path = f"{label}/imports/{relative.as_posix()}"
                if not _eligible_for_main_training(label, relative):
                    skipped_ineligible.append(display_path)
                    continue
                target = (
                    train_root
                    / label
                    / "daily-corrections"
                    / local_date.isoformat()
                    / relative
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target, duplicate = _unique_destination(target, source)
                if duplicate:
                    source.unlink()
                    already_present.append(display_path)
                else:
                    shutil.move(str(source), str(target))
                    moved.append(display_path)
            except OSError as error:
                failures.append(
                    {
                        "path": f"{label}/imports/{relative.as_posix()}",
                        "reason": str(error),
                    }
                )
    return {
        "date": local_date.isoformat(),
        "moved": len(moved),
        "already_present": len(already_present),
        "skipped_ineligible": skipped_ineligible,
        "failures": failures,
    }


def build_combined_dataset(
    train_root: Path,
    correction_root: Path,
    runtime_dir: Path,
    view: Path,
    validator: Callable[[Path], bool | None] | None = None,
) -> dict:
    """Create a space-efficient, label-preserving nightly training view."""
    _replace_dataset_view(view)
    train_files, invalid_train, repaired_train = _link_labelled_root(
        train_root, view, "primary", validator
    )
    (
        correction_files,
        invalid_correction,
        repaired_correction,
    ) = _link_labelled_root(
        correction_root,
        view,
        "correction",
        validator,
        labels=tuple(label for label in CLASS_NAMES if label != "IC"),
    )
    repaired_headers = repaired_train + repaired_correction
    return {
        "train_files": train_files,
        "correction_files": correction_files,
        "files_by_label": {
            label: sum(1 for _path in (view / label).rglob("*.lig"))
            if (view / label).is_dir()
            else 0
            for label in CLASS_NAMES
        },
        # Raw feedback stays in the correction database until the operator
        # explicitly uses “添加到纠错集”. This prevents uploads or unreviewed
        # clicks from silently entering main-model training.
        "invalid_files": invalid_train + invalid_correction,
        "repaired_headers": {
            "count": len(repaired_headers),
            "examples": repaired_headers[:20],
        },
    }


def _cleanup_old_runs(runs_dir: Path, keep: int = 3) -> None:
    runs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in runs[max(1, keep):]:
        resolved = path.resolve()
        if resolved.parent != runs_dir.resolve():
            raise RuntimeError(f"拒绝清理意外的训练目录: {path}")
        shutil.rmtree(path)


def run(args) -> dict:
    repository_root = Path(args.repository_root).resolve()
    train_root = Path(args.train_data).resolve()
    correction_root = Path(args.correction_data).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    ligclassify_root = Path(args.ligclassify_root).resolve()
    export_python = Path(args.export_python).resolve()
    training_environment = Path(sys.prefix).name
    if (
        args.required_training_env
        and training_environment.casefold()
        != args.required_training_env.casefold()
    ):
        raise RuntimeError(
            f"主模型训练必须使用 conda {args.required_training_env} 环境；"
            f"当前为 {sys.executable}"
        )
    if not export_python.is_file():
        raise FileNotFoundError(f"ONNX 导出 Python 不存在: {export_python}")
    main_dir = runtime_dir / "main_model"
    work_dir = main_dir / "work"
    view = work_dir / "dataset"
    runs_dir = main_dir / "runs"
    status = MainTrainingStatus(main_dir / "status.json")
    runs_dir.mkdir(parents=True, exist_ok=True)

    with MainTrainingLock(main_dir / "training.lock"):
        status.write(
            "preparing",
            (
                "正在检查主模型训练数据"
                if args.audit_only
                else "正在去重迁移纠错集 IC 并校验主模型训练数据"
            ),
            python_executable=sys.executable,
        )
        ic_promotion = ICCorrectionPromoter(
            correction_root,
            train_root,
            runtime_dir / "ic-promotion.json",
        )
        ic_promotion_summary = (
            ic_promotion.audit() if args.audit_only else ic_promotion.promote()
        )
        if ic_promotion_summary.get("status") == "failed":
            raise RuntimeError(
                f"IC 迁移失败: {ic_promotion_summary.get('reason', '')}"
            )
        promotion_summary = (
            {
                "date": datetime.now(CHINA_TIMEZONE).date().isoformat(),
                "moved": 0,
                "already_present": 0,
                "skipped_ineligible": [],
                "failures": [],
            }
            if args.audit_only
            else promote_daily_corrections(correction_root, train_root)
        )
        if promotion_summary["failures"]:
            raise RuntimeError(
                f"有 {len(promotion_summary['failures'])} 个纠错文件无法加入训练集"
            )
        data_summary = build_combined_dataset(
            train_root,
            correction_root,
            runtime_dir,
            view,
            validator=_validate_lig_file,
        )
        if data_summary["train_files"] + data_summary["correction_files"] == 0:
            raise RuntimeError("训练集中没有可用的 .lig 文件")
        if args.audit_only:
            result = {
                "data": data_summary,
                "promotion": promotion_summary,
                "ic_promotion": ic_promotion_summary,
                "python_executable": sys.executable,
                "export_python": str(export_python),
            }
            status.write("audited", "主模型训练数据检查完成", **result)
            return result

        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = runs_dir / run_name
        status.write(
            "training",
            "正在使用合并数据重新训练主模型",
            data=data_summary,
            promotion=promotion_summary,
            ic_promotion=ic_promotion_summary,
            output=str(output),
            python_executable=sys.executable,
            export_python=str(export_python),
        )
        torch_cache = work_dir / "torch_cache"
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torch_cache))
        _work_around_broken_windows_asyncio()
        sys.path.insert(0, str(ligclassify_root))
        try:
            import train as ligclassify_train

            training_args = ligclassify_train.build_parser().parse_args([
                "--task_data", str(view),
                "--output", str(output),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--patience", str(args.patience),
                "--samples_per_epoch", str(args.samples_per_epoch),
                "--num_workers", "0",
                "--distance_weight", "0",
            ] + (["--no_amp"] if args.no_amp else []))
            metrics = ligclassify_train.run_training(training_args)
        finally:
            try:
                sys.path.remove(str(ligclassify_root))
            except ValueError:
                pass

        status.write(
            "exporting",
            "训练完成，正在校验并激活 ONNX 主模型",
            data=data_summary,
            promotion=promotion_summary,
            ic_promotion=ic_promotion_summary,
            output=str(output),
            python_executable=sys.executable,
            export_python=str(export_python),
        )
        subprocess.run(
            [
                str(export_python),
                "-m",
                "tools.export_main_model",
                "--checkpoint",
                str(output / "model.pt"),
                "--output",
                str(main_dir / "current.onnx"),
                "--metadata",
                str(main_dir / "current.json"),
                "--ligclassify-root",
                str(ligclassify_root),
            ],
            cwd=repository_root,
            check=True,
        )
        metadata = json.loads(
            (main_dir / "current.json").read_text(encoding="utf-8")
        )
        result = {
            "data": data_summary,
            "promotion": promotion_summary,
            "ic_promotion": ic_promotion_summary,
            "output": str(output),
            "best_epoch": metrics.get("best_epoch"),
            "best_score": metrics.get("best_score"),
            "task": metrics.get("task", "classification_only"),
            "model_hash": metadata["onnx_sha256"],
            "python_executable": sys.executable,
            "export_python": str(export_python),
        }
        status.write("activated", "新主模型已激活", **result)
        _cleanup_old_runs(runs_dir, keep=args.keep_runs)
        return result


def build_parser() -> argparse.ArgumentParser:
    desktop = Path.home() / "Desktop"
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=repository)
    parser.add_argument("--train-data", default=desktop / "train_data")
    parser.add_argument("--correction-data", default=desktop / "correct_data")
    parser.add_argument(
        "--runtime-dir",
        "--feedback-dir",
        dest="runtime_dir",
        default=Path(__file__).resolve().parents[1] / "runtime",
        help="LigWeb runtime directory for main-model artifacts and status",
    )
    parser.add_argument("--ligclassify-root", default=desktop / "ligClassify")
    parser.add_argument(
        "--export-python",
        default=Path.home() / "miniconda3" / "python.exe",
        help="Python interpreter with ONNX and ONNX Runtime for export parity checks",
    )
    parser.add_argument("--required-training-env", default="ligclassify")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=120000)
    parser.add_argument("--keep-runs", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate and prepare the dataset view without moving or training data",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    status_path = Path(args.runtime_dir) / "main_model" / "status.json"
    status = MainTrainingStatus(status_path)
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        status.write(
            "failed",
            f"{type(error).__name__}: {error}",
            python_executable=sys.executable,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
