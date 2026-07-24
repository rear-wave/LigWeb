"""User-session fallback scheduler for the nightly Windows training job."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CHINA_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def next_daily_22(now: datetime) -> datetime:
    local = now.astimezone(CHINA_TIMEZONE)
    candidate = local.replace(hour=22, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate


def due_slot(now: datetime, completed_slot: str | None) -> str | None:
    local = now.astimezone(CHINA_TIMEZONE)
    slot = local.strftime("%Y-%m-%d")
    if local.hour == 22 and slot != completed_slot:
        return slot
    return None


def _single_instance_mutex():
    if os.name != "nt":
        return None, True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, "Local\\LigWebMainModelScheduler")
    if not handle:
        raise OSError("cannot create scheduler mutex")
    return handle, kernel32.GetLastError() != 183


def run_scheduler(args) -> int:
    feedback_dir = Path(args.feedback_dir).resolve()
    schedule_path = feedback_dir / "main_model" / "schedule.json"
    training_log = feedback_dir / "main_model" / "scheduled-training.log"
    repository_root = Path(args.repository_root).resolve()
    completed_slot = None
    if schedule_path.is_file():
        try:
            completed_slot = json.loads(
                schedule_path.read_text(encoding="utf-8")
            ).get("completed_slot")
        except (OSError, ValueError, AttributeError):
            completed_slot = None

    while True:
        now = datetime.now(CHINA_TIMEZONE)
        slot = due_slot(now, completed_slot)
        schedule_state = {
            "active": True,
            "pid": os.getpid(),
            "completed_slot": completed_slot,
            "next_training": next_daily_22(now).isoformat(),
            "updated_at": now.isoformat(),
            "mode": "user-session",
            "python_executable": sys.executable,
        }
        _atomic_json(
            schedule_path,
            schedule_state,
        )
        if slot is not None:
            # Claim the date before launching so a failure or restart cannot
            # start a second expensive run during the same 22:00 hour.
            completed_slot = slot
            schedule_state["completed_slot"] = completed_slot
            schedule_state["next_training"] = next_daily_22(now).isoformat()
            _atomic_json(schedule_path, schedule_state)
            training_log.parent.mkdir(parents=True, exist_ok=True)
            with training_log.open("a", encoding="utf-8") as handle:
                print(
                    f"[{now.isoformat()}] nightly main-model training started",
                    file=handle,
                    flush=True,
                )
                print(f"Python: {sys.executable}", file=handle, flush=True)
                result = subprocess.run(
                    [sys.executable, "-m", "tools.train_main_model"],
                    cwd=repository_root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
                print(
                    f"[{datetime.now(CHINA_TIMEZONE).isoformat()}] "
                    f"nightly main-model training exited with {result.returncode}",
                    file=handle,
                    flush=True,
                )
            schedule_state["last_exit_code"] = result.returncode
            schedule_state["last_finished_at"] = datetime.now(
                CHINA_TIMEZONE
            ).isoformat()
            _atomic_json(schedule_path, schedule_state)
        if args.once:
            return 0
        time.sleep(max(5.0, float(args.poll_seconds)))


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    feedback = Path.home() / "Desktop" / "correct_data" / ".ligedit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=repository)
    parser.add_argument("--feedback-dir", default=feedback)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handle, first = _single_instance_mutex()
    if not first:
        if handle is not None:
            ctypes.windll.kernel32.CloseHandle(handle)
        return 0
    try:
        return run_scheduler(args)
    finally:
        if handle is not None:
            ctypes.windll.kernel32.CloseHandle(handle)


if __name__ == "__main__":
    raise SystemExit(main())
