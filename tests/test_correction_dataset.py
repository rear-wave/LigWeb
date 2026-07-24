from ligweb.correction_dataset import (
    StoredLig,
    StoredPiece,
    deduplicate_dataset,
    default_lig_name,
    reclassify_dataset,
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


def test_reclassification_splits_pieces_by_resolved_label(tmp_path, monkeypatch):
    root = tmp_path / "correct_data"
    ic_file = root / "IC" / "sample.lig"
    ncg_file = root / "NCG" / "sample.lig"
    files = {
        ic_file: StoredLig(
            ic_file,
            b"header",
            (
                StoredPiece("1", "move-to-ncg", b"a"),
                StoredPiece("2", "stay-ic", b"b"),
            ),
        ),
        ncg_file: StoredLig(
            ncg_file,
            b"header",
            (StoredPiece("3", "move-to-ic", b"c"),),
        ),
    }
    monkeypatch.setattr(
        "ligweb.correction_dataset.iter_dataset_files", lambda _root: iter(files)
    )
    monkeypatch.setattr(
        "ligweb.correction_dataset.read_stored_lig", lambda path: files[path]
    )

    result = reclassify_dataset(
        root,
        {
            "move-to-ncg": "NCG",
            "stay-ic": "IC",
            "move-to-ic": "IC",
        },
    )

    assert result["unique_pieces"] == 3
    assert result["relabeled_pieces"] == 2
    assert result["pieces_by_source"]["IC"] == 2
    assert result["pieces_by_target"]["IC"] == 2
    assert result["pieces_by_target"]["NCG"] == 1
    assert result["relabeled_by_route"] == {"IC->NCG": 1, "NCG->IC": 1}
    assert result["output_files"] == 2
