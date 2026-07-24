"""Conservative, NumPy-only correction index for waveform classifications."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import BadZipFile

import numpy as np


CLASS_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
MIN_SUPPORT = 3
MIN_AGREEMENT = 0.80
MIN_VALIDATION_PRECISION = 0.90
LOCAL_MIN_CLASS_EXAMPLES = 2
LOCAL_MIN_SIMILARITY = 0.80
LOCAL_RADIUS_MARGIN = 0.03
LOCAL_CONFLICT_MARGIN = 0.03
CORRECTION_ALGORITHM_VERSION = "large-dataset-v3"
MAX_NEIGHBORS = 5
MAX_EXHAUSTIVE_TRAINING_ROWS = 64
MAX_VALIDATION_ROWS = 256
ARTIFACT_SCHEMA = "ligedit_correction_v1"


@dataclass(frozen=True)
class CorrectionRow:
    feature: np.ndarray
    base_label: str
    corrected_label: str


@dataclass(frozen=True)
class CorrectionDecision:
    label: str
    source: str
    similarity: float | None = None
    agreement: float | None = None
    support: int = 0


@dataclass(frozen=True)
class CorrectionIndex:
    features: np.ndarray
    base_labels: np.ndarray
    corrected_labels: np.ndarray
    base_model_hash: str
    generation: int
    threshold: float | None
    validation_precision: float
    validation_coverage: int

    @classmethod
    def empty(cls, base_model_hash: str, generation: int):
        return cls(
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype="U5"),
            np.empty(0, dtype="U5"),
            base_model_hash,
            generation,
            None,
            1.0,
            0,
        )

    @classmethod
    def from_rows(
        cls,
        rows,
        base_model_hash: str,
        generation: int,
        threshold: float | None,
        *,
        validation_precision: float = 1.0,
        validation_coverage: int = 0,
    ):
        rows = tuple(rows)
        if not rows:
            return cls.empty(base_model_hash, generation)

        features = [_normalize_feature(row.feature) for row in rows]
        feature_dimension = features[0].shape[0]
        if any(feature.shape[0] != feature_dimension for feature in features):
            raise ValueError("correction features must have a consistent dimension")
        for row in rows:
            _validate_label(row.base_label, "base label")
            _validate_label(row.corrected_label, "corrected label")

        return cls(
            np.stack(features).astype(np.float32, copy=False),
            np.asarray([row.base_label for row in rows], dtype="U5"),
            np.asarray([row.corrected_label for row in rows], dtype="U5"),
            base_model_hash,
            generation,
            threshold,
            float(validation_precision),
            int(validation_coverage),
        )


def _validate_label(label: str, description: str) -> None:
    if label not in CLASS_NAMES:
        raise ValueError(f"invalid {description}: {label!r}")


def _normalize_feature(feature) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("correction features must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(value)):
        raise ValueError("correction features must contain only finite values")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("correction features must have non-zero L2 norm")
    return value / norm


def _local_cluster_specs(index: CorrectionIndex):
    """Build conservative per-label neighborhoods from repeated corrections."""
    specs = []
    pairs = sorted(set(zip(
        index.base_labels.tolist(), index.corrected_labels.tolist()
    )))
    for base_label, corrected_label in pairs:
        positions = np.flatnonzero(
            (index.base_labels == base_label)
            & (index.corrected_labels == corrected_label)
        )
        if len(positions) < LOCAL_MIN_CLASS_EXAMPLES:
            continue
        threshold = max(
            LOCAL_MIN_SIMILARITY,
            float(index.threshold or LOCAL_MIN_SIMILARITY),
        )
        specs.append((base_label, corrected_label, positions, threshold))
    return specs


def _resolve_local_cluster(base_label, feature, index):
    """Correct near repeated examples while rejecting class-boundary ambiguity."""
    try:
        query = _normalize_feature(feature)
    except (TypeError, ValueError):
        return CorrectionDecision(base_label, "base")

    matching = np.flatnonzero(index.base_labels == base_label)
    if matching.size == 0:
        return CorrectionDecision(base_label, "base")
    all_similarities = index.features[matching] @ query
    nearest_position = matching[int(np.argmax(all_similarities))]
    nearest_label = str(index.corrected_labels[nearest_position])

    candidates = []
    for cluster_base, corrected_label, positions, threshold in _local_cluster_specs(index):
        if cluster_base != base_label:
            continue
        similarities = index.features[positions] @ query
        support = int(np.count_nonzero(similarities >= threshold))
        best_similarity = float(np.max(similarities))
        if support < LOCAL_MIN_CLASS_EXAMPLES or best_similarity < threshold:
            continue
        candidates.append((best_similarity, corrected_label, support))

    if not candidates:
        return CorrectionDecision(base_label, "base")
    candidates.sort(reverse=True)
    best_similarity, corrected_label, support = candidates[0]
    if corrected_label != nearest_label:
        return CorrectionDecision(base_label, "base")
    if (
        len(candidates) > 1
        and best_similarity - candidates[1][0] < LOCAL_CONFLICT_MARGIN
    ):
        return CorrectionDecision(base_label, "base")
    return CorrectionDecision(
        corrected_label,
        "adapter",
        similarity=best_similarity,
        agreement=1.0,
        support=support,
    )


def resolve_correction(
    base_label: str,
    feature,
    exact_label: str | None,
    suppressed: bool,
    index: CorrectionIndex | None,
) -> CorrectionDecision:
    if exact_label is not None:
        return CorrectionDecision(exact_label, "manual_exact")
    if suppressed or index is None or index.threshold is None:
        return CorrectionDecision(base_label, "base")

    def local_fallback():
        if index.validation_coverage <= 0:
            return CorrectionDecision(base_label, "base")
        return _resolve_local_cluster(base_label, feature, index)

    try:
        normalized = _normalize_feature(feature)
    except (TypeError, ValueError):
        return local_fallback()
    if index.features.ndim != 2 or index.features.shape[1] != normalized.shape[0]:
        return local_fallback()

    matching = np.flatnonzero(index.base_labels == base_label)
    if matching.size == 0:
        return local_fallback()
    similarities = index.features[matching] @ normalized
    retained = similarities >= index.threshold
    matching = matching[retained]
    similarities = similarities[retained]
    if matching.size == 0:
        return local_fallback()

    order = np.lexsort((matching, -similarities))[:MAX_NEIGHBORS]
    matching = matching[order]
    similarities = similarities[order]
    labels = index.corrected_labels[matching]
    weights = np.maximum(similarities, 0.0)

    vote_weights = {
        label: float(weights[labels == label].sum()) for label in CLASS_NAMES
    }
    winner = CLASS_NAMES[0]
    for label in CLASS_NAMES[1:]:
        if vote_weights[label] > vote_weights[winner]:
            winner = label
    support = int(np.count_nonzero(labels == winner))
    total_weight = float(weights.sum())
    agreement = vote_weights[winner] / total_weight if total_weight > 0.0 else 0.0
    if support < MIN_SUPPORT or agreement < MIN_AGREEMENT:
        return local_fallback()
    return CorrectionDecision(
        winner,
        "adapter",
        similarity=float(similarities[0]),
        agreement=agreement,
        support=support,
    )


def _sample_validation_rows(rows):
    rows = tuple(rows)
    if len(rows) <= MAX_VALIDATION_ROWS:
        return rows
    positions = np.linspace(
        0, len(rows) - 1, num=MAX_VALIDATION_ROWS, dtype=np.int64
    )
    return tuple(rows[int(position)] for position in positions)


def _large_dataset_threshold(index: CorrectionIndex) -> float:
    """Estimate a conservative threshold without exhaustive threshold search."""
    nearest_consistent = []
    pairs = sorted(set(zip(
        index.base_labels.tolist(), index.corrected_labels.tolist()
    )))
    for base_label, corrected_label in pairs:
        positions = np.flatnonzero(
            (index.base_labels == base_label)
            & (index.corrected_labels == corrected_label)
        )
        if len(positions) < LOCAL_MIN_CLASS_EXAMPLES:
            continue
        features = index.features[positions]
        pairwise = features @ features.T
        np.fill_diagonal(pairwise, -np.inf)
        nearest_consistent.extend(np.max(pairwise, axis=1).tolist())

    if not nearest_consistent:
        return LOCAL_MIN_SIMILARITY
    representative_similarity = float(np.percentile(nearest_consistent, 5))
    return float(np.clip(
        representative_similarity - LOCAL_RADIUS_MARGIN,
        LOCAL_MIN_SIMILARITY,
        1.0,
    ))


def build_candidate(rows, base_model_hash: str, generation: int) -> CorrectionIndex:
    rows = tuple(rows)
    base_index = CorrectionIndex.from_rows(
        rows, base_model_hash, generation, threshold=None
    )
    if len(rows) < 2:
        return base_index

    if len(rows) > MAX_EXHAUSTIVE_TRAINING_ROWS:
        threshold = _large_dataset_threshold(base_index)
        trial = CorrectionIndex.from_rows(
            rows,
            base_model_hash,
            generation,
            threshold=threshold,
            validation_coverage=1,
        )
        validation_rows = _sample_validation_rows(rows)
        correct, incorrect = _evaluate_leave_one_out(trial, validation_rows)
        coverage = correct + incorrect
        precision = correct / coverage if coverage else 0.0
        if coverage == 0 or precision < MIN_VALIDATION_PRECISION:
            return base_index
        return CorrectionIndex.from_rows(
            rows,
            base_model_hash,
            generation,
            threshold=threshold,
            validation_precision=precision,
            validation_coverage=coverage,
        )

    pairwise = base_index.features @ base_index.features.T
    candidates = pairwise[np.triu_indices(len(rows), k=1)]
    candidates = np.unique(np.clip(candidates[np.isfinite(candidates)], 0.0, 1.0))

    best = None
    for threshold in candidates[::-1]:
        trial = CorrectionIndex.from_rows(
            rows, base_model_hash, generation, threshold=float(threshold)
        )
        correct, incorrect = _evaluate_leave_one_out(trial, rows)
        coverage = correct + incorrect
        if coverage == 0:
            continue
        precision = correct / coverage
        if precision < MIN_VALIDATION_PRECISION:
            continue
        rank = (coverage, precision, float(threshold))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), precision, coverage)

    if best is None:
        local_trial = CorrectionIndex.from_rows(
            rows,
            base_model_hash,
            generation,
            threshold=LOCAL_MIN_SIMILARITY,
            validation_coverage=1,
        )
        correct, incorrect = _evaluate_leave_one_out(local_trial, rows)
        coverage = correct + incorrect
        precision = correct / coverage if coverage else 0.0
        if coverage == 0 or precision < MIN_VALIDATION_PRECISION:
            return base_index
        return CorrectionIndex.from_rows(
            rows,
            base_model_hash,
            generation,
            threshold=LOCAL_MIN_SIMILARITY,
            validation_precision=precision,
            validation_coverage=coverage,
        )
    _, threshold, precision, coverage = best
    return CorrectionIndex.from_rows(
        rows,
        base_model_hash,
        generation,
        threshold=threshold,
        validation_precision=precision,
        validation_coverage=coverage,
    )


def _index_without_row(index: CorrectionIndex, row: CorrectionRow) -> CorrectionIndex:
    try:
        feature = _normalize_feature(row.feature)
    except (TypeError, ValueError):
        return index
    if index.features.ndim != 2 or index.features.shape[1:] != feature.shape:
        return index

    matching = np.flatnonzero(index.base_labels == row.base_label)
    matching = [
        position
        for position in matching
        if np.array_equal(index.features[position], feature)
    ]
    if not matching:
        return index
    keep = np.ones(index.features.shape[0], dtype=bool)
    keep[matching[0]] = False
    return CorrectionIndex(
        features=index.features[keep],
        base_labels=index.base_labels[keep],
        corrected_labels=index.corrected_labels[keep],
        base_model_hash=index.base_model_hash,
        generation=index.generation,
        threshold=index.threshold,
        validation_precision=index.validation_precision,
        validation_coverage=index.validation_coverage,
    )


def _evaluate_leave_one_out(index: CorrectionIndex, rows) -> tuple[int, int]:
    correct = 0
    incorrect = 0
    for row in rows:
        evaluation_index = _index_without_row(index, row)
        decision = resolve_correction(
            row.base_label, row.feature, None, False, evaluation_index
        )
        if decision.source != "adapter":
            continue
        if decision.label == row.corrected_label:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect


def is_candidate_acceptable(
    candidate: CorrectionIndex,
    active: CorrectionIndex | None,
    rows,
    *,
    all_records_cancelled: bool = False,
) -> bool:
    rows = tuple(rows)
    try:
        _validate_index(candidate)
    except (TypeError, ValueError):
        return False

    if all_records_cancelled:
        return (
            not rows
            and candidate.threshold is None
            and candidate.features.shape[0] == 0
        )

    validation_rows = _sample_validation_rows(rows)

    compatible_active = active
    if active is not None:
        try:
            _validate_index(active)
        except (TypeError, ValueError):
            compatible_active = None
        else:
            if active.base_model_hash != candidate.base_model_hash:
                compatible_active = None
            elif (
                active.features.shape[0]
                and candidate.features.shape[0]
                and active.features.shape[1] != candidate.features.shape[1]
            ):
                compatible_active = None

    if candidate.threshold is None:
        if compatible_active is None:
            return True
        active_correct, active_incorrect = _evaluate_leave_one_out(
            compatible_active, validation_rows
        )
        active_coverage = active_correct + active_incorrect
        active_precision = (
            active_correct / active_coverage if active_coverage else 1.0
        )
        return active_precision < MIN_VALIDATION_PRECISION

    candidate_correct, candidate_incorrect = _evaluate_leave_one_out(
        candidate, validation_rows
    )
    candidate_coverage = candidate_correct + candidate_incorrect
    if candidate_coverage == 0:
        return False
    if candidate_correct / candidate_coverage < MIN_VALIDATION_PRECISION:
        return False

    if compatible_active is None:
        active_correct = active_incorrect = 0
    else:
        active_correct, active_incorrect = _evaluate_leave_one_out(
            compatible_active, validation_rows
        )
    return (
        candidate_incorrect <= active_incorrect
        and candidate_correct >= active_correct
    )


def _validate_index(index: CorrectionIndex) -> None:
    if not isinstance(index, CorrectionIndex):
        raise TypeError("expected a CorrectionIndex")
    if not isinstance(index.base_model_hash, str):
        raise ValueError("base model hash must be a string")
    if not isinstance(index.generation, int) or index.generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if index.threshold is not None:
        if not np.isfinite(index.threshold) or not 0.0 <= index.threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
    if index.features.dtype != np.float32 or index.features.ndim != 2:
        raise ValueError("features must be a two-dimensional float32 array")
    if not np.all(np.isfinite(index.features)):
        raise ValueError("features must contain only finite values")
    row_count = index.features.shape[0]
    if index.base_labels.shape != (row_count,):
        raise ValueError("base label shape does not match features")
    if index.corrected_labels.shape != (row_count,):
        raise ValueError("corrected label shape does not match features")
    if index.base_labels.dtype.kind != "U" or index.corrected_labels.dtype.kind != "U":
        raise ValueError("labels must use fixed-width Unicode arrays")
    if not set(index.base_labels.tolist()).issubset(CLASS_NAMES):
        raise ValueError("artifact contains an invalid base label")
    if not set(index.corrected_labels.tolist()).issubset(CLASS_NAMES):
        raise ValueError("artifact contains an invalid corrected label")
    if row_count:
        norms = np.linalg.norm(index.features, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("artifact features must be L2-normalized")
    if not np.isfinite(index.validation_precision):
        raise ValueError("validation precision must be finite")
    if not 0.0 <= index.validation_precision <= 1.0:
        raise ValueError("validation precision must be between zero and one")
    if (
        not isinstance(index.validation_coverage, int)
        or index.validation_coverage < 0
    ):
        raise ValueError("validation coverage must be a non-negative integer")


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    _write_bytes_fsync(temporary, payload)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_generation(root, index: CorrectionIndex) -> Path:
    _validate_index(index)
    root = Path(root)
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    name = f"generation-{index.generation:06d}"
    final_dir = models_root / name
    temporary_dir = models_root / f"{name}.tmp"
    if final_dir.exists():
        raise FileExistsError(f"generation already exists: {final_dir}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()

    try:
        adapter_path = temporary_dir / "adapter.npz"
        np.savez(
            adapter_path,
            features=index.features,
            base_labels=index.base_labels,
            corrected_labels=index.corrected_labels,
        )
        with adapter_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        metadata = {
            "schema": ARTIFACT_SCHEMA,
            "base_model_hash": index.base_model_hash,
            "generation": index.generation,
            "threshold": index.threshold,
            "metrics": {
                "validation_precision": index.validation_precision,
                "validation_coverage": index.validation_coverage,
            },
            "feature_dimension": index.features.shape[1],
            "adapter_sha256": _sha256_file(adapter_path),
        }
        metadata_payload = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8")
        _write_bytes_fsync(temporary_dir / "metadata.json", metadata_payload)
        os.replace(temporary_dir, final_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return final_dir


def _load_generation(generation_dir: Path) -> CorrectionIndex:
    metadata_path = generation_dir / "metadata.json"
    adapter_path = generation_dir / "adapter.npz"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if metadata.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported correction artifact schema")
    if metadata.get("adapter_sha256") != _sha256_file(adapter_path):
        raise ValueError("correction adapter checksum mismatch")

    with np.load(adapter_path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "base_labels", "corrected_labels"}:
            raise ValueError("correction adapter has unexpected arrays")
        features = archive["features"].copy()
        base_labels = archive["base_labels"].copy()
        corrected_labels = archive["corrected_labels"].copy()

    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("correction metadata metrics are invalid")
    index = CorrectionIndex(
        features=features,
        base_labels=base_labels,
        corrected_labels=corrected_labels,
        base_model_hash=metadata["base_model_hash"],
        generation=metadata["generation"],
        threshold=metadata["threshold"],
        validation_precision=metrics["validation_precision"],
        validation_coverage=metrics["validation_coverage"],
    )
    _validate_index(index)
    if metadata.get("feature_dimension") != features.shape[1]:
        raise ValueError("correction feature dimension mismatch")
    expected_name = f"generation-{index.generation:06d}"
    if generation_dir.name != expected_name:
        raise ValueError("correction generation directory mismatch")
    return index


def activate_generation(root, generation_dir) -> None:
    root = Path(root)
    models_root = (root / "models").resolve()
    generation_dir = Path(generation_dir).resolve()
    if generation_dir.parent != models_root or generation_dir.name.endswith(".tmp"):
        raise ValueError("generation directory must be a completed local artifact")
    index = _load_generation(generation_dir)

    root.mkdir(parents=True, exist_ok=True)
    active_path = root / "active.json"
    backup_path = root / "backup.json"
    if active_path.exists():
        _atomic_write_bytes(backup_path, active_path.read_bytes())
    pointer = {
        "schema": ARTIFACT_SCHEMA,
        "generation": index.generation,
        "directory": generation_dir.relative_to(root.resolve()).as_posix(),
    }
    payload = json.dumps(pointer, sort_keys=True, indent=2).encode("utf-8")
    _atomic_write_bytes(active_path, payload)


def load_active_index(
    root, base_model_hash: str | None = None
) -> CorrectionIndex | None:
    root = Path(root)
    try:
        pointer = json.loads((root / "active.json").read_text("utf-8"))
        if pointer.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("unsupported correction pointer schema")
        generation = pointer["generation"]
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("invalid active correction generation")
        expected_relative = f"models/generation-{generation:06d}"
        if pointer.get("directory") != expected_relative:
            raise ValueError("invalid active correction directory")
        generation_dir = (root / expected_relative).resolve()
        if generation_dir.parent != (root / "models").resolve():
            raise ValueError("active correction directory escapes the model root")
        index = _load_generation(generation_dir)
        if base_model_hash is not None and index.base_model_hash != base_model_hash:
            raise ValueError("correction artifact uses a different base model")
        return index
    except (BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError):
        return None
