"""Training-time comparison of the two arms, extrapolated from one measured epoch each.

Run from $RESULTS_DIR/R1-q3 once both run_timing_probe.sbatch tasks are done: it scans
Results/timing_probe/logs/celect_timing_*.out (relative to the working directory) for the
TIMING_ONLY line of each arm whose seed and epoch count match TIMING_SEED (default 18) and
TIMING_EPOCHS (default 1), scales seconds_per_epoch to FULL_EPOCHS (default 200), and prints the
estimated full training times, time saved, percent reduction, speed-up and the logs it used. Exits
with an error if either arm's log is missing. The real runs use early stopping and usually train
for fewer than 200 epochs.
"""
import os
import re
from pathlib import Path

seed = os.environ.get("TIMING_SEED", "18")
epochs = os.environ.get("TIMING_EPOCHS", "1")
full_epochs = int(os.environ.get("FULL_EPOCHS", "200"))

records = {}

for path in sorted(Path("Results/timing_probe/logs").glob("celect_timing_*.out")):
    text = path.read_text(errors="ignore")
    match = re.search(
        r"TIMING_ONLY setting=(\S+) seed=(\d+) epochs=(\d+) "
        r"train_seconds=([0-9.]+) seconds_per_epoch=([0-9.]+)",
        text,
    )
    if not match:
        continue

    setting, log_seed, log_epochs, train_sec, sec_per_epoch = match.groups()
    if log_seed == seed and log_epochs == epochs:
        records[setting] = {
            "path": path,
            "train_sec": float(train_sec),
            "sec_per_epoch": float(sec_per_epoch),
        }

missing = {"raw_expression", "sciLaMA"} - set(records)
if missing:
    raise SystemExit(f"Missing timing logs for: {sorted(missing)}")

raw_full = records["raw_expression"]["sec_per_epoch"] * full_epochs
sci_full = records["sciLaMA"]["sec_per_epoch"] * full_epochs
saved = raw_full - sci_full
percent_saved = 100 * saved / raw_full
speedup = raw_full / sci_full

print(f"Seed: {seed}")
print(f"Measured epochs: {epochs}")
print(f"Extrapolated epochs: {full_epochs}")
print()
print(
    "Raw expression: "
    f"{records['raw_expression']['sec_per_epoch']:.3f} sec/epoch "
    f"-> {raw_full / 60:.2f} min"
)
print(
    "sciLaMA:         "
    f"{records['sciLaMA']['sec_per_epoch']:.3f} sec/epoch "
    f"-> {sci_full / 60:.2f} min"
)
print()
print(f"Estimated time saved: {saved / 60:.2f} min")
print(f"Percent reduction: {percent_saved:.1f}%")
print(f"Speedup: {speedup:.2f}x")
print()
print("Logs:")
print(f"  raw_expression: {records['raw_expression']['path']}")
print(f"  sciLaMA:        {records['sciLaMA']['path']}")
