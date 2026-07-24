import numpy as np

from ligweb import inference
from ligweb.feedback_store import FeedbackStore
from ligweb.feedback_training import run_feedback_training


def test_clean_feedback_store_is_noop(tmp_path):
    FeedbackStore(tmp_path / "feedback.sqlite3").mark_clean(0)

    outcome = run_feedback_training(tmp_path)

    assert outcome.status == "no_changes"


def test_clean_feedback_retrains_after_main_model_changes(tmp_path, monkeypatch):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    waveform = np.arange(64, dtype=np.uint16)
    store.upsert_feedback(
        waveform,
        "a.lig",
        0,
        "160101000000.0",
        "base-a",
        "NCG",
        0.7,
        (0.1, 0.7, 0.1, 0.05, 0.05),
        "PCG",
    )
    store.mark_clean(1)
    monkeypatch.setattr(
        "ligweb.feedback_training.get_base_model_hash", lambda: "base-b"
    )
    monkeypatch.setattr(
        "ligweb.feedback_training.classify_batch_detailed",
        lambda _waveforms, daylights: [
            inference.BasePrediction(
                "NCG",
                0.7,
                (0.1, 0.7, 0.1, 0.05, 0.05),
                np.array([1.0, 0.0], dtype=np.float32),
            )
        ],
    )

    outcome = run_feedback_training(tmp_path)

    assert outcome.status != "no_changes"


def test_all_cancelled_feedback_builds_empty_generation(tmp_path, monkeypatch):
    feedback_dir = tmp_path / "feedback"
    correction_model_dir = tmp_path / "runtime"
    store = FeedbackStore(feedback_dir / "feedback.sqlite3")
    record = store.upsert_feedback(
        np.arange(64, dtype=np.uint16),
        "a.lig",
        0,
        "t",
        "base-a",
        "NCG",
        0.7,
        (0.1, 0.7, 0.1, 0.05, 0.05),
        "PCG",
    )
    store.cancel_feedback(record.waveform_hash)
    monkeypatch.setattr(
        "ligweb.feedback_training.get_base_model_hash", lambda: "base-a"
    )

    outcome = run_feedback_training(feedback_dir, correction_model_dir)

    assert outcome.status == "activated"
    assert outcome.record_count == 0
    assert store.is_dirty() is False
    assert (correction_model_dir / "active.json").is_file()
    assert not (feedback_dir / "active.json").exists()


def test_enabled_feedback_rebuilds_with_bundled_onnx(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "_session", None)
    monkeypatch.setattr(inference, "_class_names", None)
    monkeypatch.setattr(inference, "_model_hash", None)
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    waveforms = [
        (np.arange(16_000, dtype=np.uint16) + offset).astype(np.uint16)
        for offset in range(4)
    ]
    predictions = inference.classify_batch_detailed(waveforms)
    for index, (waveform, prediction) in enumerate(zip(waveforms, predictions)):
        corrected = "PCG" if prediction.label != "PCG" else "NCG"
        store.upsert_feedback(
            waveform,
            "synthetic.lig",
            index,
            str(index),
            inference.get_base_model_hash(),
            prediction.label,
            prediction.confidence,
            prediction.probabilities,
            corrected,
        )

    outcome = run_feedback_training(tmp_path)

    assert outcome.status == "activated"
    assert outcome.record_count == 4
    assert (tmp_path / "active.json").is_file()
