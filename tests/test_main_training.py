import os
from datetime import date, datetime
import struct

from tools.train_main_model import (
    CHINA_TIMEZONE,
    _validate_lig_file,
    build_combined_dataset,
    promote_daily_corrections,
)


def test_combined_dataset_uses_synced_ic_only_once(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    feedback = correction / ".ligedit"
    (train / "IC").mkdir(parents=True)
    (correction / "IC").mkdir(parents=True)
    (train / "IC" / "train.lig").write_bytes(b"train")
    (correction / "IC" / "correct.lig").write_bytes(b"correct")
    view = feedback / "main_model" / "work" / "dataset"

    summary = build_combined_dataset(train, correction, feedback, view)

    assert summary == {
        "train_files": 1,
        "correction_files": 0,
        "files_by_label": {
            "IC": 1,
            "NCG": 0,
            "NNBE": 0,
            "PCG": 0,
            "PNBE": 0,
        },
        "invalid_files": [],
        "repaired_headers": {"count": 0, "examples": []},
    }
    train_link = view / "IC" / "primary" / "train.lig"
    assert train_link.read_bytes() == b"train"
    assert not (view / "IC" / "correction" / "correct.lig").exists()
    if os.name == "nt":
        assert train_link.stat().st_ino == (train / "IC" / "train.lig").stat().st_ino


def test_combined_dataset_includes_non_ic_file_without_distance_label(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    feedback = correction / ".ligedit"
    (train / "IC").mkdir(parents=True)
    (train / "PCG" / "unknown-distance").mkdir(parents=True)
    (correction / "PCG" / "100-200km").mkdir(parents=True)
    (train / "IC" / "train.lig").write_bytes(b"train")
    (train / "PCG" / "unknown-distance" / "skip.lig").write_bytes(b"skip")
    (correction / "PCG" / "100-200km" / "keep.lig").write_bytes(b"keep")
    view = feedback / "main_model" / "work" / "dataset"

    summary = build_combined_dataset(train, correction, feedback, view)

    assert summary["train_files"] == 2
    assert summary["correction_files"] == 1
    assert (view / "PCG" / "primary" / "unknown-distance" / "skip.lig").is_file()
    assert (view / "PCG" / "correction" / "100-200km" / "keep.lig").is_file()


def test_combined_dataset_skips_and_reports_malformed_lig(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    feedback = correction / ".ligedit"
    (train / "IC").mkdir(parents=True)
    valid = train / "IC" / "valid.lig"
    valid.write_bytes(bytes(112))
    invalid = train / "IC" / "truncated.lig"
    invalid.write_bytes(b"broken")
    view = feedback / "main_model" / "work" / "dataset"

    summary = build_combined_dataset(
        train, correction, feedback, view, validator=_validate_lig_file
    )

    assert summary["train_files"] == 1
    assert len(summary["invalid_files"]) == 1
    assert summary["invalid_files"][0]["path"] == "IC/truncated.lig"
    assert not (view / "IC" / "primary" / "truncated.lig").exists()


def test_lig_validator_repairs_stale_piece_count_without_changing_mtime(tmp_path):
    path = tmp_path / "stale-count.lig"
    header = bytearray(112)
    struct.pack_into("<i", header, 4, 512)
    path.write_bytes(header + bytes(32208))
    timestamp = datetime(
        2026, 7, 20, 12, 0, tzinfo=CHINA_TIMEZONE
    ).timestamp()
    os.utime(path, (timestamp, timestamp))
    original_mtime = path.stat().st_mtime_ns

    assert _validate_lig_file(path) is True

    assert struct.unpack_from("<i", path.read_bytes(), 4)[0] == 1
    assert path.stat().st_mtime_ns == original_mtime
    assert _validate_lig_file(path) is False


def test_promote_daily_corrections_moves_all_today_classified_data(tmp_path):
    correction = tmp_path / "correct_data"
    train = tmp_path / "train_data"
    today = date(2026, 7, 21)
    today_timestamp = datetime(
        2026, 7, 21, 15, 30, tzinfo=CHINA_TIMEZONE
    ).timestamp()
    yesterday_timestamp = datetime(
        2026, 7, 20, 15, 30, tzinfo=CHINA_TIMEZONE
    ).timestamp()

    ic_today = correction / "IC" / "imports" / "source" / "today.lig"
    ic_old = correction / "IC" / "imports" / "source" / "old.lig"
    unknown_distance = (
        correction / "PCG" / "imports" / "unknown-distance" / "today.lig"
    )
    eligible_distance = (
        correction / "PCG" / "imports" / "100-200km" / "today.lig"
    )
    for path in (ic_today, ic_old, unknown_distance, eligible_distance):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    os.utime(ic_today, (today_timestamp, today_timestamp))
    os.utime(ic_old, (yesterday_timestamp, yesterday_timestamp))
    os.utime(unknown_distance, (today_timestamp, today_timestamp))
    os.utime(eligible_distance, (today_timestamp, today_timestamp))

    summary = promote_daily_corrections(correction, train, today)

    assert summary["moved"] == 3
    assert summary["failures"] == []
    assert summary["skipped_ineligible"] == []
    assert not ic_today.exists()
    assert ic_old.exists()
    assert not unknown_distance.exists()
    assert not eligible_distance.exists()
    assert (
        train
        / "IC"
        / "daily-corrections"
        / "2026-07-21"
        / "source"
        / "today.lig"
    ).is_file()
    assert (
        train
        / "PCG"
        / "daily-corrections"
        / "2026-07-21"
        / "unknown-distance"
        / "today.lig"
    ).is_file()
    assert (
        train
        / "PCG"
        / "daily-corrections"
        / "2026-07-21"
        / "100-200km"
        / "today.lig"
    ).is_file()
