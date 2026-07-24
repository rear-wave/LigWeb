import struct

import numpy as np

from ligweb.correction_dataset import read_stored_lig
from ligweb.ic_sync import ICDatasetMirror


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


def _hashes(root):
    result = set()
    for path in root.rglob("*.lig"):
        result.update(piece.digest for piece in read_stored_lig(path).pieces)
    return result


def test_ic_mirror_replaces_training_data_and_retains_correction_files(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    status = tmp_path / "runtime" / "ic-mirror.json"
    first = np.arange(16_000, dtype=np.uint16)
    second = first + 7
    obsolete = first + 14
    primary = correction / "IC" / "reviewed.lig"
    duplicate = correction / "IC" / "nested" / "duplicate.lig"
    _write_lig(primary, [first, second])
    _write_lig(duplicate, [first])
    _write_lig(train / "IC" / "obsolete.lig", [obsolete])
    source_bytes = {
        primary: primary.read_bytes(),
        duplicate: duplicate.read_bytes(),
    }

    result = ICDatasetMirror(correction, train, status).synchronize()

    assert result["status"] == "synced"
    assert result["source_files"] == 2
    assert result["source_pieces"] == 3
    assert result["copied_files"] == 2
    assert result["copied_pieces"] == 2
    assert result["duplicate_pieces"] == 1
    assert result["cleared_files"] == 1
    assert result["cleared_pieces"] == 1
    assert result["source_retained"] is True
    assert not (train / "IC" / "obsolete.lig").exists()
    assert len(_hashes(train / "IC")) == 2
    for path, payload in source_bytes.items():
        assert path.read_bytes() == payload


def test_ic_mirror_audit_does_not_change_either_dataset(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    target = train / "IC" / "existing.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])
    _write_lig(target, [np.arange(16_000, dtype=np.uint16) + 7])
    source_payload = source.read_bytes()
    target_payload = target.read_bytes()

    result = ICDatasetMirror(correction, train).audit()

    assert result["status"] == "audited"
    assert result["copied_pieces"] == 1
    assert source.read_bytes() == source_payload
    assert target.read_bytes() == target_payload


def test_ic_mirror_keeps_old_training_data_when_staging_write_fails(
    tmp_path, monkeypatch
):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    target = train / "IC" / "existing.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])
    _write_lig(target, [np.arange(16_000, dtype=np.uint16) + 7])
    source_payload = source.read_bytes()
    target_payload = target.read_bytes()

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ligweb.ic_sync.write_stored_lig", fail_write)
    result = ICDatasetMirror(correction, train).synchronize()

    assert result["status"] == "failed"
    assert "disk full" in result["reason"]
    assert source.read_bytes() == source_payload
    assert target.read_bytes() == target_payload


def test_ic_mirror_rolls_back_when_directory_install_fails(tmp_path, monkeypatch):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    target = train / "IC" / "existing.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])
    _write_lig(target, [np.arange(16_000, dtype=np.uint16) + 7])
    source_payload = source.read_bytes()
    target_payload = target.read_bytes()

    from ligweb import ic_sync

    original_replace = ic_sync.os.replace
    target_root = (train / "IC").resolve()

    def fail_staging_install(source_path, target_path):
        source_path = ic_sync.Path(source_path)
        target_path = ic_sync.Path(target_path)
        if (
            source_path.name.startswith(".ligweb-ic-staging-")
            and target_path.resolve() == target_root
        ):
            raise OSError("install failed")
        return original_replace(source_path, target_path)

    monkeypatch.setattr(ic_sync.os, "replace", fail_staging_install)
    result = ICDatasetMirror(correction, train).synchronize()

    assert result["status"] == "failed"
    assert "install failed" in result["reason"]
    assert source.read_bytes() == source_payload
    assert target.read_bytes() == target_payload


def test_ic_mirror_keeps_old_training_data_if_correction_changes(
    tmp_path, monkeypatch
):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    target = train / "IC" / "existing.lig"
    first = np.arange(16_000, dtype=np.uint16)
    second = first + 7
    _write_lig(source, [first])
    _write_lig(target, [second])
    target_payload = target.read_bytes()

    from ligweb import ic_sync

    original_write = ic_sync.write_stored_lig

    def write_then_change(*args, **kwargs):
        original_write(*args, **kwargs)
        _write_lig(source, [first, second])

    monkeypatch.setattr(ic_sync, "write_stored_lig", write_then_change)
    result = ICDatasetMirror(correction, train).synchronize()

    assert result["status"] == "failed"
    assert "纠错集 IC 发生变化" in result["reason"]
    assert target.read_bytes() == target_payload
    assert len(read_stored_lig(source).pieces) == 2


def test_ic_mirror_replaces_malformed_old_training_file(tmp_path):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    source = correction / "IC" / "pending.lig"
    malformed = train / "IC" / "broken.lig"
    _write_lig(source, [np.arange(16_000, dtype=np.uint16)])
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_bytes(b"broken")

    result = ICDatasetMirror(correction, train).synchronize()

    assert result["status"] == "synced"
    assert result["cleared_invalid_files"] == 1
    assert not malformed.exists()
    assert len(_hashes(train / "IC")) == 1
    assert source.exists()
