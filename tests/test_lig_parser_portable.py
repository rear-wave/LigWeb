import struct

from ligweb.lig_parser import ReadLigFileWithOffsets


def _gps_time(year, month, day, hour, minute, second, fraction):
    return struct.pack(
        "<bb2x6i4xd",
        1,
        1,
        year,
        month,
        day,
        hour,
        minute,
        second,
        fraction,
    )


def test_global_header_uses_fixed_little_endian_widths(tmp_path):
    path = tmp_path / "empty.lig"
    header = struct.pack(
        "<iiIIIIii",
        1001,
        0,
        2_403_235,
        2_403_283,
        0,
        74_594,
        207,
        1,
    )
    payload = header + _gps_time(2017, 7, 29, 7, 32, 6, 0.5) * 2
    assert len(payload) == 112
    path.write_bytes(payload)

    parsed, pieces, raw, offsets, header_size = ReadLigFileWithOffsets(path)

    assert parsed["version"] == 1001
    assert parsed["firstPieceCacheCount"] == 2_403_235
    assert parsed["lastPieceCacheCount"] == 2_403_283
    assert parsed["LastPieceCachePlace"] == 74_594
    assert parsed["StationID"] == 1
    assert pieces == []
    assert offsets == []
    assert raw == payload
    assert header_size == 112
