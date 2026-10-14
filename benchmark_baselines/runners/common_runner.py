#!/usr/bin/env python3
"""Shared launcher behind the run_*.py scripts in this folder.

Maps each model family and task to its training script and to the family's summary script (paths relative to
benchmark_baselines/), then runs them one at a time, each in a fresh Python process (python_bin) with
benchmark_baselines/ as the working directory. That process runs PORTABLE_LAUNCHER, a short `python -c` program
that reads the script, replaces LEGACY_REPO_ROOT with REPO_ROOT in its source, puts the script's folder and the
working directory on sys.path and executes it as __main__ with the forwarded arguments. DATA_DIR / RESULTS_DIR
(and BIOPOINTNET_ROOT / CLOUDPRED_PATH where needed) are inherited from the environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
# The scripts take their paths from DATA_DIR / RESULTS_DIR, so LEGACY_REPO_ROOT never occurs in them and the
# launcher's substitution leaves them unchanged.
REPO_ROOT = CODE_ROOT.parent
LEGACY_REPO_ROOT = "/path/to/original/CELECTION"

PORTABLE_LAUNCHER = """
from pathlib import Path
import sys

script = Path(sys.argv[1]).resolve()
legacy_root = sys.argv[2]
repo_root = sys.argv[3]
script_args = sys.argv[4:]

source = script.read_text()
source = source.replace(legacy_root, repo_root)

sys.argv = [str(script), *script_args]
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(Path.cwd()))

globals_dict = {
    "__name__": "__main__",
    "__file__": str(script),
    "__package__": None,
}
exec(compile(source, str(script), "exec"), globals_dict)
""".strip()

MODEL_TO_DATASET_SCRIPTS = {
    "celection": {
        "cardio": "celection/cardio.py",
        "cardio_biosample": "celection/cardio_biosample.py",
        "covid": "celection/covid.py",
        "lupus": "celection/lupus.py",
        "lupus_race": "celection/lupus_race.py",
    },
    "protocell4p": {
        "cardio": "protocell4p/cardio.py",
        "cardio_biosample": "protocell4p/cardio_biosample.py",
        "covid": "protocell4p/covid.py",
        "lupus": "protocell4p/lupus.py",
        "lupus_race": "protocell4p/lupus_race.py",
    },
    "deepset": {
        "cardio": "deepset/cardio.py",
        "cardio_biosample": "deepset/cardio_biosample.py",
        "covid": "deepset/covid.py",
        "lupus": "deepset/lupus.py",
        "lupus_race": "deepset/lupus_race.py",
    },
    "cloudpred": {
        "cardio": "cloudpred/cardio.py",
        "cardio_biosample": "cloudpred/cardio_biosample.py",
        "covid": "cloudpred/covid.py",
        "lupus": "cloudpred/lupus.py",
        "lupus_race": "cloudpred/lupus_race.py",
    },
    "ct_pseudobulk": {
        "cardio": "ct_pseudobulk/cardio.py",
        "cardio_biosample": "ct_pseudobulk/cardio_biosample.py",
        "covid": "ct_pseudobulk/covid.py",
        "lupus": "ct_pseudobulk/lupus.py",
        "lupus_race": "ct_pseudobulk/lupus_race.py",
    },
    "ct_proportion": {
        "cardio": "ct_proportion/cardio.py",
        "cardio_biosample": "ct_proportion/cardio_biosample.py",
        "covid": "ct_proportion/covid.py",
        "lupus": "ct_proportion/lupus.py",
        "lupus_race": "ct_proportion/lupus_race.py",
    },
    "pseudobulk": {
        "cardio": "pseudobulk/cardio.py",
        "cardio_biosample": "pseudobulk/cardio_biosample.py",
        "covid": "pseudobulk/covid.py",
        "lupus": "pseudobulk/lupus.py",
        "lupus_race": "pseudobulk/lupus_race.py",
    },
}

SUMMARY_SCRIPT_BY_MODEL = {
    "celection": "celection/summarize.py",
    "protocell4p": "protocell4p/summarize.py",
    "deepset": "deepset/summarize.py",
    "cloudpred": "cloudpred/summarize.py",
    "ct_pseudobulk": "ct_pseudobulk/summarize_linear_baselines.py",
    "ct_proportion": "ct_pseudobulk/summarize_linear_baselines.py",
    "pseudobulk": "ct_pseudobulk/summarize_linear_baselines.py",
}


def datasets_for_model(model_name: str) -> list[str]:
    """Sorted task names available for a model family."""
    return sorted(MODEL_TO_DATASET_SCRIPTS[model_name].keys())


def script_for(model_name: str, dataset_name: str) -> str:
    """Training script (relative to benchmark_baselines/) of one model family and task; ValueError if unknown."""
    available = MODEL_TO_DATASET_SCRIPTS[model_name]
    if dataset_name not in available:
        choices = ", ".join(sorted(available))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}' for model '{model_name}'. Choices: {choices}"
        )
    return available[dataset_name]


def run_rel_script(
    rel_script: str,
    python_bin: str,
    dry_run: bool,
    script_args: list[str] | None = None,
) -> None:
    """Print the launch command for a script (relative to benchmark_baselines/) and run it unless dry_run.

    Raises FileNotFoundError for a missing script and CalledProcessError when the script fails.
    """
    script_path = CODE_ROOT / rel_script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [
        python_bin,
        "-c",
        PORTABLE_LAUNCHER,
        str(script_path),
        LEGACY_REPO_ROOT,
        str(REPO_ROOT),
        *(script_args or []),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return

    subprocess.run(cmd, check=True, cwd=CODE_ROOT)


def run_model(
    model_name: str,
    dataset_name: str,
    python_bin: str,
    dry_run: bool,
    summary: bool,
) -> None:
    """Training script for one task (or every task if dataset_name is "all"), then the summary if requested."""
    targets = datasets_for_model(model_name) if dataset_name == "all" else [dataset_name]

    for ds in targets:
        run_rel_script(script_for(model_name, ds), python_bin=python_bin, dry_run=dry_run)

    if summary:
        summary_script = SUMMARY_SCRIPT_BY_MODEL.get(model_name)
        if summary_script:
            run_rel_script(summary_script, python_bin=python_bin, dry_run=dry_run)
