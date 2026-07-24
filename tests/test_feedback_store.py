import csv
import numpy as np

from ligweb.feedback_store import FeedbackStore, waveform_digest


PROBS = (0.05, 0.70, 0.05, 0.15, 0.05)


def test_upsert_deduplicates_and_preserves_created_time(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    waveform = np.arange(16000, dtype=np.uint16)
    first = store.upsert_feedback(
        waveform=waveform,
        source_name="a.lig",
        piece_index=3,
        event_time="2026-07-20T01:02:03Z",
        base_model_hash="base-a",
        base_label="NCG",
        base_confidence=0.70,
        probabilities=PROBS,
        corrected_label="PCG",
    )
    second = store.upsert_feedback(
        waveform=waveform.copy(),
        source_name="copy.lig",
        piece_index=8,
        event_time="2026-07-20T01:02:03Z",
        base_model_hash="base-a",
        base_label="NCG",
        base_confidence=0.70,
        probabilities=PROBS,
        corrected_label="PNBE",
    )
    assert first.waveform_hash == waveform_digest(waveform)
    assert second.created_at == first.created_at
    assert second.corrected_label == "PNBE"
    assert len(store.list_records()) == 1
    assert store.count_records() == 1
    assert store.count_records(enabled_only=True) == 1
    np.testing.assert_array_equal(second.waveform, waveform)
    assert store.is_dirty()


def test_cancel_suppresses_then_upsert_reenables(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    waveform = np.arange(32, dtype=np.uint16)
    record = store.upsert_feedback(
        waveform, "a.lig", 0, "t", "base-a", "NCG", 0.7, PROBS, "PCG"
    )
    assert store.cancel_feedback(record.waveform_hash)
    assert store.get_record(record.waveform_hash).enabled is False
    assert store.count_records(enabled_only=True) == 0
    store.upsert_feedback(
        waveform, "a.lig", 0, "t", "base-a", "NCG", 0.7, PROBS, "IC"
    )
    assert store.get_record(record.waveform_hash).enabled is True


def test_export_omits_waveform_blob_and_absolute_path(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    record = store.upsert_feedback(
        np.arange(8, dtype=np.uint16), "C:/private/data/a.lig", 1, "t", "m", "NCG", 0.7,
        PROBS, "PCG"
    )
    assert record.source_name == "a.lig"
    output = tmp_path / "feedback.csv"
    store.export_csv(output)
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["source_name"] == "a.lig"
    assert "waveform_blob" not in row


def test_list_records_isolates_corrupt_blob(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    record = store.upsert_feedback(
        np.arange(8, dtype=np.uint16), "a.lig", 0, "t", "m", "NCG", 0.7,
        PROBS, "PCG"
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE feedback_records SET waveform_blob=? WHERE waveform_hash=?",
            (b"broken", record.waveform_hash),
        )
    records, failures = store.list_records_with_failures()
    assert records == []
    assert failures == [(record.waveform_hash, "invalid compressed waveform")]


def test_list_records_isolates_non_blob_and_invalid_probabilities(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    first = store.upsert_feedback(
        np.arange(8, dtype=np.uint16), "a.lig", 0, "t", "m", "NCG", 0.7,
        PROBS, "PCG"
    )
    second = store.upsert_feedback(
        np.arange(9, dtype=np.uint16), "a.lig", 1, "t", "m", "NCG", 0.7,
        PROBS, "PCG"
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE feedback_records SET waveform_blob=? WHERE waveform_hash=?",
            ("not-a-blob", first.waveform_hash),
        )
        connection.execute(
            "UPDATE feedback_records SET probabilities=? WHERE waveform_hash=?",
            ("null", second.waveform_hash),
        )
    records, failures = store.list_records_with_failures()
    assert records == []
    assert failures == [
        (first.waveform_hash, "invalid compressed waveform"),
        (second.waveform_hash, "invalid probabilities"),
    ]


def test_mark_clean_records_generation(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.mark_clean(4)
    assert store.is_dirty() is False
    assert store.get_state("active_generation") == "4"
    assert store.get_state("training_revision") == "similarity-change-v4"
