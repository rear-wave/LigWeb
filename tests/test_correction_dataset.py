from ligweb.correction_dataset import (
    StoredLig,
    StoredPiece,
    deduplicate_dataset,
    default_lig_name,
)


def test_generated_import_name_returns_to_original_style():
    assert (
        default_lig_name("GZ_170820072459.2144324_reviewed_102219dcc502.lig")
        == "GZ_170820072459.2144324.lig"
    )
    assert (
        default_lig_name("GZ_170820072459.2144324_corrected_d3fccb5cbb01.lig")
        == "GZ_170820072459.2144324.lig"
    )
    assert default_lig_name("GZ_170820072459.2144324.lig") == (
        "GZ_170820072459.2144324.lig"
    )


def test_deduplication_removes_duplicate_pieces(tmp_path, monkeypatch):
    root = tmp_path / "correct_data"
    first = root / "IC" / "imports" / "sample_corrected_aaaaaaaaaaaa.lig"
    second = root / "IC" / "imports" / "sample_reviewed_bbbbbbbbbbbb.lig"
    files = {
        first: StoredLig(
            first,
            b"header",
            (
                StoredPiece("1", "duplicate", b"a"),
                StoredPiece("2", "manual", b"b"),
            ),
        ),
        second: StoredLig(
            second,
            b"header",
            (
                StoredPiece("1", "duplicate", b"a"),
                StoredPiece("3", "model", b"c"),
            ),
        ),
    }
    monkeypatch.setattr(
        "ligweb.correction_dataset.iter_dataset_files", lambda _root: iter(files)
    )
    monkeypatch.setattr(
        "ligweb.correction_dataset.read_stored_lig", lambda path: files[path]
    )

    result = deduplicate_dataset(root, apply=False)

    assert result["input_pieces"] == 4
    assert result["unique_pieces"] == 3
    assert result["duplicate_pieces_removed"] == 1
    assert result["output_files"] == 1
