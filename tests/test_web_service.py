from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from ligweb.inference import BasePrediction, PredictionResult
from ligweb.feedback_store import waveform_digest
from ligweb.config import LigWebConfig
from ligweb.service import LigDocument, LigWebService, _decimate


def _config(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    train.mkdir()
    correction.mkdir()
    return LigWebConfig(
        repository_root=tmp_path,
        train_data_dir=train,
        correction_data_dir=correction,
        feedback_dir=tmp_path / "runtime",
        model_dir=tmp_path / "runtime",
        exports_dir=tmp_path / "runtime" / "exports",
        max_cached_files=2,
    )


def _document(path):
    waveforms = [
        np.arange(32, dtype=np.uint16),
        np.arange(32, dtype=np.uint16) + 7,
    ]
    pieces = [
        (
            f"16010100000{index}.0000000",
            {
                "0": waveform,
                "m_samplingRate": 5_000_000.0,
                "m_numOfData": len(waveform),
                "m_numOfChannel": 1,
                "m_stationID": 1,
            },
        )
        for index, waveform in enumerate(waveforms)
    ]
    raw = bytearray(16) + b"aaa" + b"bbb"
    return LigDocument(
        path=path,
        mtime_ns=1,
        header={"version": 1001, "NumOfPiece": 2},
        pieces=pieces,
        raw_data=bytes(raw),
        piece_offsets=[(16, 19), (19, 22)],
        header_size=16,
        base_predictions=[
            BasePrediction(
                "NNBE",
                0.8,
                (0.03, 0.05, 0.8, 0.07, 0.05),
                np.array([1.0, float(index)], dtype=np.float32),
            )
            for index in range(2)
        ],
    )


def _empty_lig_payload():
    header = struct.pack(
        "<iiIIIIii", 1001, 0, 1, 1, 0, 0, 1, 1
    )
    gps_time = struct.pack(
        "<bb2x6i4xd", 1, 1, 2026, 7, 20, 12, 0, 0, 0.0
    )
    return header + gps_time * 2


def test_file_listing_is_paginated_and_confined_to_dataset(tmp_path):
    config = _config(tmp_path)
    (config.train_data_dir / "IC").mkdir()
    (config.train_data_dir / "IC" / "a.lig").write_bytes(b"one")
    (config.train_data_dir / "IC" / "b.lig").write_bytes(b"two")
    service = LigWebService(config)

    page = service.list_files("train", query="IC/", offset=0, limit=1)
    assert page["total"] == 2
    assert len(page["files"]) == 1
    assert page["files"][0]["path"].startswith("IC/")
    with pytest.raises(PermissionError):
        service._resolve_dataset_file("train", "../outside.lig")


def test_close_document_releases_cached_file(tmp_path):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    service._cache[source.resolve()] = _document(source.resolve())

    assert service.close_document("train", "sample.lig") is True
    assert source.resolve() not in service._cache
    assert service.close_document("train", "sample.lig") is False


def test_save_document_removes_marked_piece_without_backup(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    document = _document(source.resolve())
    source.write_bytes(document.raw_data)
    document.mtime_ns = source.stat().st_mtime_ns
    service = LigWebService(config)
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)

    def read_saved(path):
        raw = Path(path).read_bytes()
        return (
            {"NumOfPiece": 1},
            [document.pieces[1]],
            raw,
            [(16, len(raw))],
            16,
        )

    monkeypatch.setattr("ligweb.service.ReadLigFileWithOffsets", read_saved)

    result = service.save_document("train", "sample.lig", [0])

    assert result["deleted_count"] == 1
    assert result["piece_count"] == 1
    assert result["backup_path"] is None
    assert source.read_bytes()[16:] == b"bbb"
    assert int.from_bytes(source.read_bytes()[4:8], "little", signed=True) == 1
    assert not source.with_suffix(".lig.bak").exists()


def test_upload_stays_in_inbox_until_explicitly_added(tmp_path):
    config = _config(tmp_path)
    service = LigWebService(config)

    first = service.upload_lig("external.lig", _empty_lig_payload())
    second = service.upload_lig("external.lig", _empty_lig_payload())

    assert first["dataset"] == "inbox"
    assert first["path"] == "external.lig"
    assert second["path"] == "external_2.lig"
    assert service.list_files("inbox")["total"] == 2
    assert service.list_files("correction")["total"] == 0
    assert (config.inbox_dir / "external.lig").is_file()


def test_feedback_uses_external_correction_database(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr("ligweb.service.get_base_model_hash", lambda: "base-a")

    result = service.save_feedback("train", "sample.lig", 0, "IC")

    assert result["corrected_label"] == "IC"
    assert result["base_label"] == "NNBE"
    assert (config.feedback_dir / "feedback.sqlite3").is_file()
    assert service.training_status()["record_count"] == 1
    assert service.training_status()["inference_revision"] == 1


def test_piece_summary_exposes_main_and_correction_results(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    prediction = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="IC",
        source="adapter",
        correction_similarity=0.94,
    )
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [prediction, prediction]
    )

    classification = service.list_pieces("train", "sample.lig")["pieces"][0][
        "classification"
    ]
    assert classification["main_model"]["label"] == "NNBE"
    assert classification["correction_model"] == {
        "label": "IC",
        "applied": True,
        "source": "adapter",
        "similarity": 0.94,
    }


def test_correction_folder_label_overrides_non_manual_model_result(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    source = config.correction_data_dir / "IC" / "sample.lig"
    source.parent.mkdir()
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    prediction = PredictionResult(
        base_label="NCG",
        base_confidence=0.8,
        probabilities=(0.03, 0.8, 0.05, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="NCG",
        source="base",
    )
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [prediction, prediction]
    )

    classification = service.list_pieces(
        "correction", "IC/sample.lig"
    )["pieces"][0]["classification"]

    assert classification["main_model"]["label"] == "NCG"
    assert classification["correction_model"] == {
        "label": "IC",
        "applied": True,
        "source": "dataset_label",
        "similarity": None,
    }


def test_model_runtime_is_separate_from_correction_dataset(tmp_path):
    config = _config(tmp_path)

    assert config.feedback_dir == tmp_path / "runtime"
    assert config.main_model_dir == tmp_path / "runtime" / "main_model"
    assert config.correction_model_dir == tmp_path / "runtime"
    assert config.inbox_dir == tmp_path / "runtime" / "inbox"
    assert config.exports_dir == tmp_path / "runtime" / "exports"


def test_reviewed_import_uses_manual_and_model_results(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    manual = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="IC",
        source="manual_exact",
    )
    model = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([0.0, 1.0], dtype=np.float32),
        effective_label="PCG",
        source="adapter",
        correction_similarity=0.94,
    )
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [manual, model]
    )
    monkeypatch.setattr("ligweb.service.waveform_index", lambda _root: set())

    def write_lig(path, _header, pieces):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(len(pieces).to_bytes(4, "little"))

    monkeypatch.setattr("ligweb.service.write_stored_lig", write_lig)

    result = service.import_corrected_pieces(
        "train", "sample.lig", [0, 1]
    )

    assert result["imported_piece_count"] == 2
    assert result["manual_piece_count"] == 1
    assert result["model_piece_count"] == 1
    assert result["duplicate_skipped_count"] == 0
    assert result["skipped_indices"] == []
    assert result["source_removed"] is False
    assert result["files"] == [
        {
            "label": "IC",
            "path": "IC/sample.lig",
            "piece_count": 1,
            "size": 4,
            "main_training_eligible": True,
        },
        {
            "label": "PCG",
            "path": "PCG/sample.lig",
            "piece_count": 1,
            "size": 4,
            "main_training_eligible": True,
        },
    ]
    for item in result["files"]:
        output = config.correction_data_dir / item["path"]
        assert output.is_file()
        assert int.from_bytes(output.read_bytes(), "little") == 1


def test_correction_import_accepts_piece_without_manual_feedback(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    prediction = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="IC",
        source="adapter",
        correction_similarity=0.91,
    )
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [prediction, prediction]
    )
    monkeypatch.setattr("ligweb.service.waveform_index", lambda _root: set())

    def write_lig(path, _header, pieces):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(len(pieces).to_bytes(4, "little"))

    monkeypatch.setattr("ligweb.service.write_stored_lig", write_lig)

    result = service.import_corrected_pieces("train", "sample.lig", [0])

    assert result["imported_piece_count"] == 1
    assert result["manual_piece_count"] == 0
    assert result["model_piece_count"] == 1
    assert result["duplicate_skipped_count"] == 0
    assert result["skipped_indices"] == []
    assert result["files"][0]["label"] == "IC"
    assert result["files"][0]["path"] == "IC/sample.lig"


def test_correction_import_skips_waveform_already_in_dataset(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    prediction = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="IC",
        source="adapter",
    )
    digest = waveform_digest(document.pieces[0][1]["0"])
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [prediction, prediction]
    )
    monkeypatch.setattr("ligweb.service.waveform_index", lambda _root: {digest})

    result = service.import_corrected_pieces("train", "sample.lig", [0])

    assert result["files"] == []
    assert result["imported_piece_count"] == 0
    assert result["duplicate_skipped_count"] == 1
    assert result["duplicate_skipped_indices"] == [0]


def test_completed_inbox_lig_is_removed_after_import(tmp_path, monkeypatch):
    config = _config(tmp_path)
    service = LigWebService(config)
    source = config.inbox_dir / "sample.lig"
    source.write_bytes(b"source")
    document = _document(source)
    prediction = PredictionResult(
        base_label="NNBE",
        base_confidence=0.8,
        probabilities=(0.03, 0.05, 0.8, 0.07, 0.05),
        feature=np.array([1.0, 0.0], dtype=np.float32),
        effective_label="IC",
        source="adapter",
    )
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)
    monkeypatch.setattr(
        service, "_effective_predictions", lambda _document: [prediction, prediction]
    )
    monkeypatch.setattr("ligweb.service.waveform_index", lambda _root: set())

    def write_lig(path, _header, pieces):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(len(pieces).to_bytes(4, "little"))

    monkeypatch.setattr("ligweb.service.write_stored_lig", write_lig)

    result = service.import_corrected_pieces("inbox", "sample.lig", [0])

    assert result["imported_piece_count"] == 1
    assert result["source_removed"] is True
    assert not source.exists()


def test_export_keeps_selected_pieces_under_external_export_dir(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)

    result = service.export_pieces(
        "train", "sample.lig", [1], "selected.lig"
    )

    output = config.exports_dir / "selected.lig"
    assert result["piece_count"] == 1
    assert output.read_bytes()[16:] == b"bbb"
    assert int.from_bytes(output.read_bytes()[4:8], "little", signed=True) == 1


def test_daynight_export_builds_zip_with_one_lig_per_period(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    source = config.train_data_dir / "sample.lig"
    source.touch()
    service = LigWebService(config)
    document = _document(source)
    document.pieces[0] = ("160101000000.0000000", document.pieces[0][1])
    document.pieces[1] = ("160101120000.0000000", document.pieces[1][1])
    monkeypatch.setattr(service, "_load_document", lambda *_args: document)

    result = service.export_by_daynight(
        "train", "sample.lig", [], "sample_daynight.zip"
    )

    archive_path = config.exports_dir / "sample_daynight.zip"
    assert result["piece_count"] == 2
    assert result["day_count"] == 1
    assert result["night_count"] == 1
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"sample_day.lig", "sample_night.lig"}
        assert archive.read("sample_day.lig")[16:] == b"aaa"
        assert archive.read("sample_night.lig")[16:] == b"bbb"


def test_envelope_decimation_keeps_peak_and_requested_bound():
    values = np.zeros(10_000)
    values[5432] = 99
    positions, samples = _decimate(values, 400)
    assert len(samples) <= 400
    assert 5432 in positions
    assert 99 in samples


def test_fastapi_factory_registers_api_and_static_routes(tmp_path):
    from ligweb.app import create_app

    application = create_app(_config(tmp_path))
    paths = {route.path for route in application.routes}
    assert "/api/health" in paths
    assert "/api/feedback" in paths
    assert "/api/training" in paths
    assert "/api/ic-promotion" in paths
    assert "/api/correction-imports" in paths
    assert "/api/export/daynight" in paths
    assert "/api/files/{dataset}/{file_path:path}/save" in paths
    assert "/api/files/{dataset}/{file_path:path}/session" in paths
    assert "/api/files/{dataset}/{file_path:path}/piece/{piece_index}" in paths
    assert "/api/uploads" in paths
