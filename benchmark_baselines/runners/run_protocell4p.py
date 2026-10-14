#!/usr/bin/env python3
"""Same as run_celection.py for ProtoCell4P: protocell4p/*.py, then protocell4p/summarize.py.
"""
from __future__ import annotations

import argparse
import sys

from common_runner import datasets_for_model, run_model

MODEL = "protocell4p"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProtoCell4P benchmark scripts.")
    parser.add_argument("--dataset", default="all", choices=["all", *datasets_for_model(MODEL)])
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ProtoCell4P summary script after training runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model(MODEL, args.dataset, python_bin=args.python, dry_run=args.dry_run, summary=args.summary)


if __name__ == "__main__":
    main()
