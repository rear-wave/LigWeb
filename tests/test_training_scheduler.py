from datetime import datetime, timedelta, timezone

from ligweb.scheduler import (
    CHINA_TIMEZONE,
    CorrectionTrainingScheduler,
    next_daily_22,
)
from tools.main_training_scheduler import due_slot


def test_correction_scheduler_submits_once_at_each_top_of_hour():
    state = {}
    submissions = []
    scheduler = CorrectionTrainingScheduler(
        submissions.append,
        lambda: state.get("slot"),
        lambda value: state.__setitem__("slot", value),
    )
    due = datetime(2026, 7, 20, 14, 0, 10, tzinfo=CHINA_TIMEZONE)

    assert scheduler.tick(due) is True
    assert scheduler.tick(due.replace(second=40)) is False
    assert len(submissions) == 1
    assert scheduler.tick(due + timedelta(hours=1)) is True
    assert len(submissions) == 2


def test_correction_scheduler_waits_for_exact_hour():
    scheduler = CorrectionTrainingScheduler(
        lambda _reason: None, lambda: None, lambda _value: None
    )
    not_due = datetime(2026, 7, 20, 14, 59, tzinfo=CHINA_TIMEZONE)
    assert scheduler.tick(not_due) is False


def test_next_main_training_is_daily_at_22_china_time():
    utc = timezone.utc
    before = datetime(2026, 7, 20, 12, 0, tzinfo=utc)
    after = datetime(2026, 7, 20, 15, 0, tzinfo=utc)
    assert next_daily_22(before).isoformat() == "2026-07-20T22:00:00+08:00"
    assert next_daily_22(after).isoformat() == "2026-07-21T22:00:00+08:00"


def test_user_session_main_scheduler_claims_each_22_hour_once():
    due = datetime(2026, 7, 20, 22, 30, tzinfo=CHINA_TIMEZONE)
    assert due_slot(due, None) == "2026-07-20"
    assert due_slot(due, "2026-07-20") is None
    assert due_slot(due.replace(hour=21), None) is None
