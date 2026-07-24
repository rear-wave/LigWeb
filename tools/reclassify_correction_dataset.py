"""Physically split the correction dataset by each waveform's resolved label."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from ligweb.config import LigWebConfig
from ligweb.correction_dataset import (
    iter_dataset_files,
    read_stored_lig,
    reclassify_dataset,
)
from ligweb.service import LigWebService


def collect_resolved_labels(service: LigWebService) -> tuple[dict[str, str], dict]:
    """Resolve manual feedback first, then the active correction model."""
    root = service.config.correction_data_dir.resolve()
    labels_by_digest: dict[str, str] = {}
    sources = Counter()
    conflicts = []
    file_count = 0
    piece_count = 0

    for source_path in iter_dataset_files(root):
        file_count += 1
        relative_path = source_path.relative_to(root).as_posix()
        document = service._load_document("correction", relative_path)
        stored = read_stored_lig(source_path)
        predictions = service._effective_predictions(document)
        if len(stored.pieces) != len(predictions):
            raise ValueError(
                f"piece/prediction count mismatch for {relative_path}: "
                f"{len(stored.pieces)} != {len(predictions)}"
            )
        for piece, prediction in zip(stored.pieces, predictions):
            piece_count += 1
            previous = labels_by_digest.get(piece.digest)
            if previous is not None and previous != prediction.effective_label:
                conflicts.append(
                    {
                        "waveform_hash": piece.digest,
                        "first_label": previous,
                        "next_label": prediction.effective_label,
                        "path": relative_path,
                    }
                )
                continue
            labels_by_digest[piece.digest] = prediction.effective_label
            sources[prediction.source] += 1

    if conflicts:
        raise ValueError(
            f"{len(conflicts)} duplicate waveforms resolved to conflicting labels"
        )
    return labels_by_digest, {
        "resolved_files": file_count,
        "resolved_pieces": piece_count,
        "resolved_by_source": dict(sorted(sources.items())),
    }


def run(apply: bool) -> dict:
    config = replace(
        LigWebConfig.from_env(),
        auto_correction_training=False,
        auto_ic_sync=False,
    )
    service = LigWebService(config)
    labels_by_digest, resolution = collect_resolved_labels(service)
    workspace = config.model_dir / "dataset-maintenance"
    result = reclassify_dataset(
        config.correction_data_dir,
        labels_by_digest,
        apply=apply,
        workspace_root=workspace,
    )
    result.update(resolution)
    result["time"] = datetime.now(timezone.utc).isoformat()

    workspace.mkdir(parents=True, exist_ok=True)
    report_path = workspace / "last-reclassification.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["report_path"] = str(report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace the five active class directories after staged validation",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
