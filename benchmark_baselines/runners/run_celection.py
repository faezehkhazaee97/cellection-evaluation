#!/usr/bin/env python3
"""Runs the CELLECTION training scripts (celection/*.py) for one task or all five, then celection/summarize.py.

Options: --dataset {all,cardio,cardio_biosample,covid,lupus,lupus_race} (default all), --python (interpreter for
the scripts, default the current one), --summary / --no-summary (default on) and --dry-run (print the commands
only). The work is done by common_runner.run_model(); see README.md here.
"""
from __future__ import annotations

import argparse
import sys

from common_runner import datasets_for_model, run_model

MODEL = "celection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CELECTION benchmark scripts.")
    parser.add_argument("--dataset", default="all", choices=["all", *datasets_for_model(MODEL)])
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CELECTION summary script after training runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model(MODEL, args.dataset, python_bin=args.python, dry_run=args.dry_run, summary=args.summary)


if __name__ == "__main__":
    main()
