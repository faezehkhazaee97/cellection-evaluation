#!/usr/bin/env python3
"""Runs a single script of benchmark_baselines/ through the common launcher, e.g. from benchmark_baselines/:

    python runners/run_raw_script.py plot/combine_all_test_summaries.py

The path is relative to benchmark_baselines/; further arguments (optionally after `--`) are forwarded to the
script. Options: --python (interpreter, default the current one) and --dry-run (print the command only).
"""
from __future__ import annotations

import argparse
import sys

from common_runner import run_rel_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run any benchmark_baselines script through the portable launcher."
    )
    parser.add_argument("script", help="Relative path under benchmark_baselines/, e.g. plot/combine_all_test_summaries.py")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args, script_args = parser.parse_known_args()
    args.script_args = script_args
    return args


def main() -> None:
    args = parse_args()
    forwarded_args = args.script_args
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    run_rel_script(
        args.script,
        python_bin=args.python,
        dry_run=args.dry_run,
        script_args=forwarded_args,
    )


if __name__ == "__main__":
    main()
