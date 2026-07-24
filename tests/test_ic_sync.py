import struct

import numpy as np

from ligweb.correction_dataset import read_stored_lig
from ligweb.ic_sync import ICDataSynchronizer


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
        + b"".join(_piece(values, 8 + index) for index, values in enumerate(waveforms))
    )


def test_ic_sync_skips_manually_synced_waveforms_and_removes_stale_managed_data(
    tmp_path,
):
    train = tmp_path / "train_data"
    correction = tmp_path / "correct_data"
    status = correction / ".ligedit" / "ic-sync.json"
    first = np.arange(16_000, dtype=np.uint16)
    second = first + 7
    source = correction / "IC" / "reviewed.lig"
    _write_lig(train / "IC" / "manual.lig", [first])
    _write_lig(source, [first, second])
    synchronizer = ICDataSynchronizer(correction, train, status)

    initial = synchronizer.sync(force=True)

    assert initial["status"] == "synced"
    assert initial["source_pieces"] == 2
    assert initial["skipped_existing_pieces"] == 1
    assert initial["synced_pieces"] == 1
    managed = train / "IC" / "_ligweb_sync" / "reviewed.lig"
    assert len(read_stored_lig(managed).pieces) == 1

    _write_lig(source, [first])
    updated = synchronizer.sync(force=True)

    assert updated["synced_pieces"] == 0
    assert updated["removed_synced_pieces"] == 1
    assert not managed.exists()
