import hashlib
import json

import numpy as np
import pytest

from ligweb.correction_model import (
    CorrectionIndex,
    CorrectionRow,
    build_candidate,
    resolve_correction,
)


def _unit(x, y):
    value = np.array([x, y], dtype=np.float32)
    return value / np.linalg.norm(value)


def test_exact_and_suppressed_states_precede_adapter():
    index = CorrectionIndex.empty("base-a", generation=1)
    exact = resolve_correction("NCG", _unit(1, 0), "PCG", False, index)
    suppressed = resolve_correction("NCG", _unit(1, 0), None, True, index)
    assert (exact.label, exact.source) == ("PCG", "manual_exact")
    assert (suppressed.label, suppressed.source) == ("NCG", "base")


def test_three_consistent_neighbors_override_matching_base_class():
    rows = [
        CorrectionRow(_unit(1, 0.00), "NCG", "PCG"),
        CorrectionRow(_unit(1, 0.01), "NCG", "PCG"),
        CorrectionRow(_unit(1, -0.01), "NCG", "PCG"),
        CorrectionRow(_unit(0.99, 0.02), "NCG", "PCG"),
    ]
    index = CorrectionIndex.from_rows(rows, "base-a", 1, threshold=0.95)
    decision = resolve_correction("NCG", _unit(1, 0), None, False, index)
    assert decision.label == "PCG"
    assert decision.source == "adapter"
    assert decision.support >= 3
    assert decision.agreement >= 0.80


def test_different_base_class_and_split_votes_fall_back():
    rows = [
        CorrectionRow(_unit(1, 0.00), "NCG", "PCG"),
        CorrectionRow(_unit(1, 0.01), "NCG", "PNBE"),
        CorrectionRow(_unit(1, -0.01), "NCG", "PCG"),
    ]
    index = CorrectionIndex.from_rows(rows, "base-a", 1, threshold=0.90)
    assert (
        resolve_correction("NNBE", _unit(1, 0), None, False, index).source
        == "base"
    )
    assert (
        resolve_correction("NCG", _unit(1, 0), None, False, index).source
        == "base"
    )


def test_index_normalizes_valid_features():
    index = CorrectionIndex.from_rows(
        [CorrectionRow(np.array([3.0, 4.0]), "NCG", "PCG")],
        "base-a",
        1,
        threshold=0.9,
    )
    np.testing.assert_allclose(np.linalg.norm(index.features, axis=1), [1.0])
    assert index.features.dtype == np.float32


@pytest.mark.parametrize(
    "feature",
    [
        np.array([0.0, 0.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
    ],
)
def test_index_rejects_zero_and_non_finite_rows(feature):
    with pytest.raises(ValueError):
        CorrectionIndex.from_rows(
            [CorrectionRow(feature, "NCG", "PCG")],
            "base-a",
            1,
            threshold=0.9,
        )


def test_candidate_selects_generalized_threshold_by_leave_one_out():
    rows = [
        CorrectionRow(_unit(1, 0.00), "NCG", "PCG"),
        CorrectionRow(_unit(1, 0.01), "NCG", "PCG"),
        CorrectionRow(_unit(1, -0.01), "NCG", "PCG"),
        CorrectionRow(_unit(1, 0.02), "NCG", "PCG"),
    ]
    candidate = build_candidate(rows, "base-a", generation=2)
    assert candidate.threshold is not None
    assert candidate.validation_precision == 1.0
    assert candidate.validation_coverage == 4


def test_candidate_disables_generalization_when_precision_gate_fails():
    rows = [
        CorrectionRow(_unit(1, i / 100), "NCG", "PCG" if i % 2 else "PNBE")
        for i in range(8)
    ]
    candidate = build_candidate(rows, "base-a", generation=2)
    assert candidate.threshold is None
    assert candidate.validation_coverage == 0


def test_large_candidate_uses_bounded_validation_and_generalizes():
    rows = [
        CorrectionRow(_unit(1, (index % 11 - 5) / 100), "NCG", "IC")
        for index in range(300)
    ]
    candidate = build_candidate(rows, "base-a", generation=2)
    assert candidate.threshold is not None
    assert candidate.validation_precision == 1.0
    assert 0 < candidate.validation_coverage <= 256
    decision = resolve_correction(
        "NCG", _unit(1, 0.003), None, False, candidate
    )
    assert (decision.label, decision.source) == ("IC", "adapter")


def test_small_repeated_clusters_enable_local_generalization():
    rows = [
        CorrectionRow(_unit(1, offset), "IC", "NNBE")
        for offset in (0.00, 0.01, -0.01)
    ] + [
        CorrectionRow(_unit(offset, 1), "IC", "PCG")
        for offset in (0.00, 0.01)
    ] + [
        CorrectionRow(_unit(-1, 0), "IC", "NCG")
    ]
    candidate = build_candidate(rows, "base-a", generation=2)
    assert candidate.threshold is not None
    assert candidate.validation_precision == 1.0
    assert candidate.validation_coverage > 0

    nnbe = resolve_correction("IC", _unit(1, 0.005), None, False, candidate)
    pcg = resolve_correction("IC", _unit(0.005, 1), None, False, candidate)
    singleton = resolve_correction("IC", _unit(-1, 0.005), None, False, candidate)
    assert (nnbe.label, nnbe.source) == ("NNBE", "adapter")
    assert (pcg.label, pcg.source) == ("PCG", "adapter")
    assert (singleton.label, singleton.source) == ("IC", "base")


def test_generalized_candidate_must_not_cover_fewer_correct_rows_than_active():
    from ligweb.correction_model import is_candidate_acceptable

    rows = [
        CorrectionRow(_unit(1, offset), "NCG", "PCG")
        for offset in (0.00, 0.01, -0.01, 0.02)
    ] + [
        CorrectionRow(_unit(0.00, 1 + offset), "NCG", "PCG")
        for offset in (0.00, 0.01, -0.01, 0.02)
    ]
    candidate = CorrectionIndex.from_rows(rows[:4], "base-a", 2, threshold=0.95)
    active = CorrectionIndex.from_rows(rows, "base-a", 1, threshold=0.95)
    assert is_candidate_acceptable(candidate, active, rows) is False


def test_exact_only_candidate_replacement_is_conservative():
    from ligweb.correction_model import is_candidate_acceptable

    rows = [
        CorrectionRow(_unit(1, offset), "NCG", "PCG")
        for offset in (0.00, 0.01, -0.01, 0.02)
    ]
    exact_only = CorrectionIndex.empty("base-a", generation=2)
    valid_active = CorrectionIndex.from_rows(rows, "base-a", 1, threshold=0.95)
    wrong_rows = [CorrectionRow(row.feature, row.base_label, "PNBE") for row in rows]
    invalid_active = CorrectionIndex.from_rows(
        wrong_rows, "base-a", 1, threshold=0.95
    )

    assert is_candidate_acceptable(exact_only, None, rows) is True
    assert is_candidate_acceptable(exact_only, valid_active, rows) is False
    assert is_candidate_acceptable(exact_only, invalid_active, rows) is True
    incompatible = CorrectionIndex.empty("base-b", generation=1)
    assert is_candidate_acceptable(exact_only, incompatible, rows) is True


def test_all_cancelled_path_accepts_empty_generation():
    from ligweb.correction_model import is_candidate_acceptable

    active_rows = [
        CorrectionRow(_unit(1, offset), "NCG", "PCG")
        for offset in (0.00, 0.01, -0.01, 0.02)
    ]
    active = CorrectionIndex.from_rows(active_rows, "base-a", 1, threshold=0.95)
    empty = CorrectionIndex.empty("base-a", generation=2)
    assert (
        is_candidate_acceptable(
            empty, active, [], all_records_cancelled=True
        )
        is True
    )


def test_artifact_activation_keeps_backup(tmp_path):
    from ligweb.correction_model import activate_generation, load_active_index, save_generation

    first = CorrectionIndex.empty("base-a", generation=1)
    second = CorrectionIndex.empty("base-a", generation=2)
    first_dir = save_generation(tmp_path, first)
    assert first_dir == tmp_path / "models" / "generation-000001"
    activate_generation(tmp_path, first_dir)
    second_dir = save_generation(tmp_path, second)
    activate_generation(tmp_path, second_dir)
    assert load_active_index(tmp_path).generation == 2
    assert json.loads((tmp_path / "backup.json").read_text())["generation"] == 1


def test_artifact_uses_non_object_arrays_and_validates_requested_model(tmp_path):
    from ligweb.correction_model import activate_generation, load_active_index, save_generation

    rows = [CorrectionRow(_unit(1, 0), "NCG", "PCG")]
    index = CorrectionIndex.from_rows(rows, "base-a", 3, threshold=0.95)
    generation_dir = save_generation(tmp_path, index)
    metadata = json.loads((generation_dir / "metadata.json").read_text("utf-8"))
    assert metadata["schema"] == "ligedit_correction_v1"
    assert metadata["feature_dimension"] == 2
    assert metadata["adapter_sha256"] == hashlib.sha256(
        (generation_dir / "adapter.npz").read_bytes()
    ).hexdigest()
    with np.load(generation_dir / "adapter.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)

    activate_generation(tmp_path, generation_dir)
    assert load_active_index(tmp_path, "base-a").generation == 3
    assert load_active_index(tmp_path, "base-b") is None


def test_loader_rejects_corrupt_checksum(tmp_path):
    from ligweb.correction_model import activate_generation, load_active_index, save_generation

    generation_dir = save_generation(
        tmp_path, CorrectionIndex.empty("base-a", generation=1)
    )
    activate_generation(tmp_path, generation_dir)
    with (generation_dir / "adapter.npz").open("ab") as stream:
        stream.write(b"corrupt")
    assert load_active_index(tmp_path) is None
