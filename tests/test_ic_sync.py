import struct

import numpy as np

from ligweb.correction_dataset import read_stored_lig
from ligweb.ic_sync import ICCorrectionPromoter


def _gps_time(hour=8):
    return struct.pack(
        "<bb2x6i4xd",
        1,
        1,
        2026,
        7,
        24,
        hour,
        0,
        0,
        0.0,
    )


def _piece(values, hour=8):
    values = np.asarray(values, dtype="<u2")
    metadata = bytearray(208)
    struct.pack_into("<i", metadata, 0, 1001)
    struct.pack_into("<I", metadata, 4, len(values))
    struct.pack_into("<d", metadata, 8, 5_000_000.0)
    struct.pack_into("<i", metadata, 20, len(values))
    struct.pack_into("<i", metadata, 24, 1)
    struct.pack_into("<i", metadata, 64, 1)
    struct.pack_into("<i", metadata, 72, len(values) * 2)
    metadata[104:144] = _gps_time(hour)
    metadata[144:184] = _gps_time(hour)
    return bytes(metadata) + values.tobytes()


def _write_lig(path, waveforms):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(112)
    struct.pack_into("<i", header, 0, 1001)
    struct.pack_into("<i", header, 4, len(waveforms))
    header[32:72] = _gps_time()
    header[72:112] = _gps_time()
    path.write_bytes(
        bytes(header)
        + b"".join(
            _piece(values, 8 + index)
            for index, values in enumerate(waveforms)
        )
    )


def test_ic_promotion_appends_only_new_waveforms_then_deletes_source(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    status = tmp_path / "runtime" / "ic-promotion.json"
    first = np.arange(16_000, dtype=np.uint16)
    second = first + 7
    target = train / "IC" / "reviewed.lig"
    source = correction / "IC" / "reviewed.lig"
    _write_lig(target, [first])
    _write_lig(source, [first, second])

    result = ICCorrectionPromoter(correction, train, status).promote()

    assert result["status"] == "promoted"
    assert result["source_pieces"] == 2
    assert result["moved_pieces"] == 1
    assert result["duplicate_pieces"] == 1
    assert result["deleted_files"] == 1
    assert not source.exists()
    assert len(read_stored_lig(target).pieces) == 2


def test_ic_promotion_deletes_source_when_all_waveforms_already_exist(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    values = np.arange(16_000, dtype=np.uint16)
    source = correction / "IC" / "duplicate.lig"
    _write_lig(train / "IC" / "existing.lig", [values])
    _write_lig(source, [values])

    result = ICCorrectionPromoter(correction, train).promote()

    assert result["moved_pieces"] == 0
    assert result["duplicate_pieces"] == 1
    assert result["deleted_files"] == 1
    assert not source.exists()
    assert not (train / "IC" / "duplicate.lig").exists()


def test_ic_promotion_audit_does_not_change_data(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])

    result = ICCorrectionPromoter(correction, train).audit()

    assert result["status"] == "audited"
    assert result["moved_pieces"] == 1
    assert result["deleted_files"] == 0
    assert source.exists()
    assert not train.exists()


def test_ic_promotion_keeps_source_when_target_write_fails(tmp_path, monkeypatch):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ligweb.ic_sync.write_stored_lig", fail_write)
    result = ICCorrectionPromoter(correction, train).promote()

    assert result["status"] == "failed"
    assert "disk full" in result["reason"]
    assert source.exists()


def test_ic_promotion_keeps_source_if_it_changes_during_write(
    tmp_path, monkeypatch
):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    first = np.arange(16_000, dtype=np.uint16)
    second = first + 7
    _write_lig(source, [first])

    from ligweb import ic_sync

    original_write = ic_sync.write_stored_lig

    def write_then_change(*args, **kwargs):
        original_write(*args, **kwargs)
        _write_lig(source, [first, second])

    monkeypatch.setattr(ic_sync, "write_stored_lig", write_then_change)
    result = ICCorrectionPromoter(correction, train).promote()

    assert result["status"] == "failed"
    assert "发生变化" in result["reason"]
    assert len(read_stored_lig(source).pieces) == 2
