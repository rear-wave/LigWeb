import numpy as np
import csv

from ligweb import inference as classify_module
from ligweb.correction_model import CorrectionIndex, CorrectionRow
from ligweb.feedback_store import FeedbackStore


class FakeSession:
    def run(self, _outputs, inputs):
        count = len(inputs["waveform"])
        logits = np.tile(
            np.array([[0.0, 3.0, 1.0, 0.5, -1.0]], np.float32),
            (count, 1),
        )
        features = np.tile(np.array([[3.0, 4.0]], np.float32), (count, 1))
        return [logits, features]


class FakeDualSession:
    def run(self, _outputs, inputs):
        assert set(inputs) == {"local", "global_view", "daylight"}
        assert inputs["local"].shape[1:] == (1, 8000)
        assert inputs["global_view"].shape[1:] == (1, 2000)
        assert inputs["daylight"].tolist() == [[1.0]]
        logits = np.array([[3.0, 0.0, 1.0, 0.5, -1.0]], np.float32)
        features = np.array([[0.0, 2.0]], np.float32)
        return [logits, features]


def test_detailed_batch_returns_probabilities_and_features(monkeypatch):
    monkeypatch.setattr(classify_module, "_session", FakeSession())
    monkeypatch.setattr(classify_module, "_class_names", list(classify_module.CLASS_NAMES))
    results = classify_module.classify_batch_detailed(
        [np.arange(16000, dtype=np.float32)]
    )
    assert results[0].label == "NCG"
    assert len(results[0].probabilities) == 5
    np.testing.assert_allclose(results[0].feature, [0.6, 0.8], atol=1e-6)


def test_detailed_batch_supports_latest_dual_view_model(monkeypatch):
    monkeypatch.setattr(classify_module, "_session", FakeDualSession())
    monkeypatch.setattr(classify_module, "_model_path", None)
    monkeypatch.setattr(classify_module, "_model_schema", "ligedit_main_model_v2")
    monkeypatch.setattr(classify_module, "_model_metadata", {
        "preprocess_config": {
            "local_length": 8000,
            "global_length": 2000,
            "use_filter": False,
            "cutoff_hz": 120000.0,
            "sample_rate_hz": 5000000.0,
        }
    })
    monkeypatch.setattr(
        classify_module, "_class_names", list(classify_module.CLASS_NAMES)
    )
    result = classify_module.classify_batch_detailed(
        [np.arange(16000, dtype=np.float32)], daylights=[True]
    )[0]
    assert result.label == "IC"
    np.testing.assert_allclose(result.feature, [0.0, 1.0], atol=1e-6)


def test_manual_correction_keeps_base_confidence():
    base = classify_module.BasePrediction(
        "NCG", 0.70, (0.1, 0.7, 0.1, 0.05, 0.05),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    result = classify_module.apply_correction(
        base,
        exact_label="PCG",
        index=CorrectionIndex.empty("base-a", 1),
        base_model_hash="base-a",
    )
    assert result.effective_label == "PCG"
    assert result.source == "manual_exact"
    assert result.base_confidence == 0.70


def test_incompatible_adapter_falls_back_to_base():
    base = classify_module.BasePrediction(
        "NCG", 0.70, (0.1, 0.7, 0.1, 0.05, 0.05),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    result = classify_module.apply_correction(
        base,
        index=CorrectionIndex.empty("other", 1),
        base_model_hash="current",
    )
    assert (result.effective_label, result.source) == ("NCG", "base")


def test_feedback_batch_applies_exact_and_adapter_labels(tmp_path):
    waveform = np.arange(16000, dtype=np.uint16)
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.upsert_feedback(
        waveform, "sample.lig", 0, "t", "base-a", "NCG", 0.7,
        (0.1, 0.7, 0.1, 0.05, 0.05), "PCG",
    )
    exact_context = classify_module.load_correction_context(tmp_path)
    base = classify_module.BasePrediction(
        "NCG", 0.70, (0.1, 0.7, 0.1, 0.05, 0.05),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    exact = classify_module.apply_feedback_batch(
        [waveform], [base], exact_context
    )[0]
    assert (exact.effective_label, exact.source) == ("PCG", "manual_exact")

    rows = [
        CorrectionRow(np.array([1.0, offset], np.float32), "NCG", "NNBE")
        for offset in (0.0, 0.01, -0.01)
    ]
    index = CorrectionIndex.from_rows(rows, "base-a", 2, threshold=0.9)
    adapter_context = classify_module.CorrectionContext({}, index, "base-a")
    adapted = classify_module.apply_feedback_batch(
        [waveform], [base], adapter_context
    )[0]
    assert (adapted.effective_label, adapted.source) == ("NNBE", "adapter")


def test_folder_classification_writes_feedback_adjusted_result(
    tmp_path, monkeypatch
):
    waveform = np.arange(16000, dtype=np.uint16)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.lig").write_bytes(b"lig")

    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.upsert_feedback(
        waveform, "sample.lig", 0, "t", "base-a", "NCG", 0.7,
        (0.1, 0.7, 0.1, 0.05, 0.05), "PCG",
    )
    context = classify_module.load_correction_context(tmp_path)
    monkeypatch.setattr(classify_module, "_session", FakeSession())
    monkeypatch.setattr(
        classify_module, "_class_names", list(classify_module.CLASS_NAMES)
    )
    monkeypatch.setattr(
        classify_module, "load_correction_context", lambda: context
    )
    monkeypatch.setattr(
        "ligweb.lig_parser.ReadLigFile", lambda _path: {"t": {"0": waveform}}
    )

    counts = classify_module.classify_folder(input_dir, output_dir, batch_size=1)
    assert counts["PCG"] == 1
    with (output_dir / "summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["predicted_class"] == "PCG"
    assert row["base_predicted_class"] == "NCG"
    assert row["prediction_source"] == "manual_exact"


def test_bundled_five_class_model_exposes_128_features(monkeypatch):
    monkeypatch.setattr(classify_module, "_session", None)
    monkeypatch.setattr(classify_module, "_class_names", None)
    monkeypatch.setattr(classify_module, "_model_hash", None)
    waveform = np.linspace(0, 4095, 16000, dtype=np.float32)
    result = classify_module.classify_batch_detailed([waveform], batch_size=1)[0]
    assert result.label in classify_module.CLASS_NAMES
    assert len(result.probabilities) == 5
    assert result.feature.shape == (128,)
    np.testing.assert_allclose(np.linalg.norm(result.feature), 1.0, atol=1e-5)
