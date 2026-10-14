#!/usr/bin/env python3
"""Same as run_celection.py for the cell-type proportion baselines (ct_proportion/<task>.py).

Its summary step, ct_pseudobulk/summarize_linear_baselines.py, is shared with the other two linear baselines.
"""
from __future__ import annotations

import argparse
import sys

from common_runner import datasets_for_model, run_model

MODEL = "ct_proportion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CT-proportion benchmark scripts.")
    parser.add_argument("--dataset", default="all", choices=["all", *datasets_for_model(MODEL)])
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CT/Proportion/Pseudobulk summary aggregation script",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model(MODEL, args.dataset, python_bin=args.python, dry_run=args.dry_run, summary=args.summary)


if __name__ == "__main__":
    main()
