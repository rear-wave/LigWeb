"""Small, dependency-free helpers for writing selected LIG pieces."""

from __future__ import annotations

from pathlib import Path
import struct


def save_lig_file(
    output_path,
    raw_data: bytes,
    header_size: int,
    piece_offsets,
    deleted_indices,
) -> None:
    """Write a LIG while omitting selected zero-based piece indices."""
    deleted = {int(index) for index in deleted_indices}
    payload = bytearray(raw_data[: int(header_size)])
    struct.pack_into("<i", payload, 4, len(piece_offsets) - len(deleted))
    for index, (start, end) in enumerate(piece_offsets):
        if index not in deleted:
            payload.extend(raw_data[int(start):int(end)])
    Path(output_path).write_bytes(payload)
