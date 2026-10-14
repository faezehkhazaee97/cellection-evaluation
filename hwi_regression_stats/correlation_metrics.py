"""How closely do predicted hand-wing index (HWI) values track the observed ones? (Figure S18)

Statistics only: the predictions come from a separate CELLECTION HWI regression run. For the
train and the test species separately, observed vs predicted HWI gets a Pearson and a Spearman
correlation with scipy's two-sided p-values and a 95% percentile bootstrap CI (5,000 resamples
of species with replacement, seed 0). A baseline that predicts the training-set mean for every
test species goes through the same function. All of it is printed, and each split gets an
observed-vs-predicted scatter plot with the identity line.

Reads $RESULTS_DIR/R2-q5/Data/all_predictions_Hand-Wing.Index.csv: one row per species, with
the unnamed index column (species name, read as sample_id), original_value, prediction and
split ("train" / "test"). Writes to $RESULTS_DIR/R2-q5/Results/:
  correlation_metrics.csv              rows train, test, test_mean_baseline; columns n, then
                                       r, p, ci_low and ci_high for pearson_ and spearman_
  scatter_train.pdf, scatter_test.pdf
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # inputs and outputs of this script


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Two-sided 95% percentile interval.
CI_PERCENTILES = [2.5, 97.5]

path = f"{RESULTS_DIR}/R2-q5/Data/all_predictions_Hand-Wing.Index.csv"
save_dir = f"{RESULTS_DIR}/R2-q5/Results"
os.makedirs(save_dir, exist_ok=True)
df = pd.read_csv(path).rename(columns={"Unnamed: 0": "sample_id"})


def corr_with_ci(y, yhat, method="pearson", n_boot=5000, seed=0):
    """Return (r, p, ci_low, ci_high); method "pearson", anything else means Spearman.

    The generator is re-created from `seed` on every call, so with the default seed the Pearson
    and Spearman CIs are built from the same resamples.
    """
    rng = np.random.default_rng(seed)
    if method == "pearson":
        r, p = stats.pearsonr(y, yhat)
        fn = lambda a, b: stats.pearsonr(a, b)[0]
    else:
        r, p = stats.spearmanr(y, yhat)
        fn = lambda a, b: stats.spearmanr(a, b)[0]

    # idx is (n_boot, n): each row resamples the n paired species with replacement.
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boot = [fn(y[i], yhat[i]) for i in idx]
    lo, hi = np.percentile(boot, CI_PERCENTILES)
    return r, p, lo, hi


results = []
for split in ["train", "test"]:
    sub = df[df["split"] == split]
    y = sub["original_value"].to_numpy()
    yhat = sub["prediction"].to_numpy()

    pr, pp, pr_lo, pr_hi = corr_with_ci(y, yhat, "pearson")
    sr, sp, sr_lo, sr_hi = corr_with_ci(y, yhat, "spearman")

    print(f"\n{split.upper()} n={len(y)}")
    print(f"Pearson r={pr:.3f}, p={pp:.3g}, 95% CI=({pr_lo:.3f},{pr_hi:.3f})")
    print(f"Spearman r={sr:.3f}, p={sp:.3g}, 95% CI=({sr_lo:.3f},{sr_hi:.3f})")
    results.append({
        "split": split,
        "n": len(y),
        "pearson_r": pr,
        "pearson_p": pp,
        "pearson_ci_low": pr_lo,
        "pearson_ci_high": pr_hi,
        "spearman_r": sr,
        "spearman_p": sp,
        "spearman_ci_low": sr_lo,
        "spearman_ci_high": sr_hi,
    })

    plt.figure(figsize=(5, 4))
    plt.scatter(y, yhat, s=12, alpha=0.6)
    lims = [
        min(np.min(y), np.min(yhat)),
        max(np.max(y), np.max(yhat))
    ]
    plt.plot(lims, lims, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Original value")
    plt.ylabel("Prediction")
    plt.title(
        f"{split} (Pearson r={pr:.2f}, p={pp:.2g})\n"
        f"(Spearman r={sr:.2f}, p={sp:.2g})",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"scatter_{split}.pdf"), dpi=300)
    plt.close()

# Baseline: every test species is predicted as the training-set mean HWI. That prediction is
# constant, so scipy returns NaN (and warns) for its r, p and CI bounds.
train_mean = df[df["split"] == "train"]["original_value"].mean()
test = df[df["split"] == "test"]
y_test = test["original_value"].to_numpy()
y_baseline = np.full_like(y_test, train_mean)

pr, pp, pr_lo, pr_hi = corr_with_ci(y_test, y_baseline, "pearson")
sr, sp, sr_lo, sr_hi = corr_with_ci(y_test, y_baseline, "spearman")
print("\nTEST baseline (predict train mean)")
print(f"Pearson r={pr:.3f}, p={pp:.3g}, 95% CI=({pr_lo:.3f},{pr_hi:.3f})")
print(f"Spearman r={sr:.3f}, p={sp:.3g}, 95% CI=({sr_lo:.3f},{sr_hi:.3f})")
results.append({
    "split": "test_mean_baseline",
    "n": len(y_test),
    "pearson_r": pr,
    "pearson_p": pp,
    "pearson_ci_low": pr_lo,
    "pearson_ci_high": pr_hi,
    "spearman_r": sr,
    "spearman_p": sp,
    "spearman_ci_low": sr_lo,
    "spearman_ci_high": sr_hi,
})

pd.DataFrame(results).to_csv(
    os.path.join(save_dir, "correlation_metrics.csv"),
    index=False
)
