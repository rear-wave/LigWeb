"""Clock-based automation for the lightweight correction trainer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from threading import Event, Thread
from typing import Callable


CHINA_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def next_hour(now: datetime) -> datetime:
    local = now.astimezone(CHINA_TIMEZONE)
    return local.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def next_daily_22(now: datetime) -> datetime:
    local = now.astimezone(CHINA_TIMEZONE)
    candidate = local.replace(hour=22, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate


class CorrectionTrainingScheduler:
    """Submit at most one correction-training job per wall-clock hour."""

    def __init__(
        self,
        submit: Callable[[str], object],
        read_slot: Callable[[], str | None],
        write_slot: Callable[[str], None],
        *,
        enabled: bool = True,
        poll_seconds: float = 15.0,
    ) -> None:
        self.submit = submit
        self.read_slot = read_slot
        self.write_slot = write_slot
        self.enabled = bool(enabled)
        self.poll_seconds = max(1.0, float(poll_seconds))
        self._stop = Event()
        self._thread: Thread | None = None
        self._logger = logging.getLogger(__name__)

    def tick(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        local = (now or datetime.now(CHINA_TIMEZONE)).astimezone(CHINA_TIMEZONE)
        if local.minute != 0:
            return False
        slot = local.strftime("%Y-%m-%dT%H")
        if self.read_slot() == slot:
            return False
        # Persist before submission so a restart in the same minute cannot
        # enqueue a duplicate job.
        self.write_slot(slot)
        self.submit(f"整点自动训练（{local:%Y-%m-%d %H:00}）")
        return True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    self._logger.exception("automatic correction training failed")
                self._stop.wait(self.poll_seconds)

        self._thread = Thread(
            target=run,
            name="ligedit-correction-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds + 1.0))


class PeriodicTaskScheduler:
    """Run one idempotent maintenance task immediately and on an interval."""

    def __init__(
        self,
        task: Callable[[], object],
        *,
        enabled: bool = True,
        poll_seconds: float = 60.0,
        thread_name: str = "ligweb-periodic-task",
    ) -> None:
        self.task = task
        self.enabled = bool(enabled)
        self.poll_seconds = max(5.0, float(poll_seconds))
        self.thread_name = str(thread_name)
        self._stop = Event()
        self._thread: Thread | None = None
        self._logger = logging.getLogger(__name__)

    def tick(self) -> bool:
        if not self.enabled:
            return False
        self.task()
        return True

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    self._logger.exception("periodic LigWeb task failed")
                self._stop.wait(self.poll_seconds)

        self._thread = Thread(
            target=run,
            name=self.thread_name,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds + 1.0))
