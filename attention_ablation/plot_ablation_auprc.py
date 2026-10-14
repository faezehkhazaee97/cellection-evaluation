"""Redraw one dataset's AUPRC ablation figures from an existing ablation_results.csv.

Plotting only. Any of the five training scripts' CSVs works (it needs seed, fold, frac_removed,
setting and AUPRC). The figures differ from the training scripts' own in three ways: the curve
has a legend, a missing std (a single run) is drawn as 0, and PNG copies are saved at 300 dpi.
Writes to --save-dir, by default the CSV's folder, replacing any files of the same name:
  ablation_auprc_curve.{pdf,png}           mean +/- std AUPRC per fraction, top vs random
  ablation_delta_auprc_barplot.{pdf,png}   delta = top - random per fraction
  ablation_pivot_AUPRC.csv                 top, random and delta per seed x fold x fraction

Example:
    python plot_ablation_auprc.py --csv $RESULTS_DIR/R2-Q4/Results/Covid/ablation_results.csv [--save-dir DIR]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_auprc_from_csv(csv_path: str, save_dir: str) -> None:
    """Write the AUPRC curve and delta bar plot (PDF and PNG) and the AUPRC pivot to save_dir."""
    df_ab = pd.read_csv(csv_path)
    required = {"seed", "fold", "frac_removed", "setting", "AUPRC"}
    missing = required.difference(df_ab.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    for setting in ["top", "random"]:
        sub = df_ab[df_ab["setting"] == setting]
        mean = sub.groupby("frac_removed")["AUPRC"].mean()
        std = sub.groupby("frac_removed")["AUPRC"].std().fillna(0)

        plt.plot(mean.index, mean.values, marker="o", label=setting)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.20)

    plt.xlabel("Fraction of cells removed")
    plt.ylabel("AUPRC")
    plt.title("AUPRC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_auprc_curve.pdf")
    plt.savefig(f"{save_dir}/ablation_auprc_curve.png", dpi=300)
    plt.close()

    pivot_auprc = (
        df_ab.pivot_table(
            index=["seed", "fold", "frac_removed"],
            columns="setting",
            values="AUPRC",
        ).reset_index()
    )
    pivot_auprc["delta"] = pivot_auprc["top"] - pivot_auprc["random"]

    plt.figure(figsize=(6, 5))
    sns.barplot(data=pivot_auprc, x="frac_removed", y="delta", color="darkorange")
    plt.axhline(0, color="black")
    plt.ylabel("ΔAUPRC (Top − Random)")
    plt.xlabel("Fraction removed")
    plt.title("AUPRC")
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_delta_auprc_barplot.pdf")
    plt.savefig(f"{save_dir}/ablation_delta_auprc_barplot.png", dpi=300)
    plt.close()

    pivot_auprc.to_csv(f"{save_dir}/ablation_pivot_AUPRC.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AUPRC ablation figures from an ablation_results.csv file."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to ablation_results.csv",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory. Defaults to CSV parent directory.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir else os.path.dirname(args.csv)
    plot_auprc_from_csv(args.csv, save_dir)
    print(f"Saved AUPRC plots and pivot CSV to: {save_dir}")


if __name__ == "__main__":
    main()
