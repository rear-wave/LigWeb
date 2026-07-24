"""Synchronize approved IC corrections into the managed training subtree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ligweb.ic_sync import ICDataSynchronizer


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
        default=desktop / "correct_data" / ".ligedit" / "ic-sync.json",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = ICDataSynchronizer(
        Path(args.correction_data),
        Path(args.train_data),
        Path(args.status),
    ).sync(force=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
