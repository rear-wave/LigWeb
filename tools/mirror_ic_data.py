"""Audit or explicitly run the nightly IC correction-data mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ligweb.ic_sync import ICDatasetMirror


def build_parser() -> argparse.ArgumentParser:
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data", default=desktop / "train_data"
    )
    parser.add_argument(
        "--correction-data", default=desktop / "correct_data"
    )
    parser.add_argument(
        "--status",
        default=Path(__file__).resolve().parents[1] / "runtime" / "ic-mirror.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace training IC; without this flag the command is read-only",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    mirror = ICDatasetMirror(
        Path(args.correction_data),
        Path(args.train_data),
        Path(args.status),
    )
    result = mirror.synchronize() if args.apply else mirror.audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
