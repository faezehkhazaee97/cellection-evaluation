"""CloudPred test metrics averaged per dataset folder of $RESULTS_DIR/R2-q1/Results/CloudPred/.

In every sub-folder, the first *_results_all_folds.csv or *_results_all_folds.pth that os.walk finds is read (a .pth
holds 'fold_<fold>_seed_<seed>' -> per-run metric table, one row per metric); folders without one are skipped. The
test auROC, macroF1 and macroAUPRC are averaged over the runs and written back into CloudPred/:
  combined_metrics.csv                     one row per dataset; columns AUROC, MacroF1, MacroAUPRC
  AUROC.csv, MacroF1.csv, MacroAUPRC.csv   one row ("CloudPred"), one column per dataset
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import pandas as pd
import torch

BASE_DIR = f"{RESULTS_DIR}/R2-q1/Results/CloudPred"
OUTPUT_BASE = BASE_DIR

datasets = {}

def find_metrics_file(folder_path):
    """Path of the first *_results_all_folds.csv / .pth file under folder_path (os.walk order), or None."""
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith("_results_all_folds.csv") or f.endswith("_results_all_folds.pth"):
                return os.path.join(root, f)
    return None


def load_metrics(path):
    """Per-run test metrics (auROC_test, macroF1_test, macroAUPRC_test columns) from a .csv or .pth results file."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df

    if path.endswith(".pth"):
        data = torch.load(path, map_location="cpu")
        # dict 'fold_<fold>_seed_<seed>' -> DataFrame with one row per metric and one value column
        rows = []
        for key, df in data.items():
            if "auROC_test" in df.index:
                rows.append({
                    "fold_seed": key,
                    "auROC_test": float(df.loc["auROC_test"].values[0]),
                    "macroF1_test": float(df.loc["macroF1_test"].values[0]),
                    "macroAUPRC_test": float(df.loc["macroAUPRC_test"].values[0]),
                })
        df = pd.DataFrame(rows)
        return df

    raise ValueError("Unsupported file type: " + path)


print("\nSearching for metric files...\n")

for dataset in os.listdir(BASE_DIR):
    dataset_dir = os.path.join(BASE_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    metrics_path = find_metrics_file(dataset_dir)
    if metrics_path is None:
        print(f"Skipping {dataset}: no metrics file found.\n")
        continue

    print(f"Loading {dataset}: {metrics_path}")
    df = load_metrics(metrics_path)

    df = df.rename(columns={
        "auROC_test": "AUROC",
        "macroF1_test": "MacroF1",
        "macroAUPRC_test": "MacroAUPRC"
    })

    datasets[dataset] = {
        "AUROC": df["AUROC"].mean(),
        "MacroF1": df["MacroF1"].mean(),
        "MacroAUPRC": df["MacroAUPRC"].mean()
    }

combined_df = pd.DataFrame(datasets).T
combined_df.index.name = "Dataset"
combined_df.to_csv(os.path.join(OUTPUT_BASE, "combined_metrics.csv"))

print("\nSaved combined metrics:")
print(os.path.join(OUTPUT_BASE, "combined_metrics.csv"))

for metric in ["AUROC", "MacroF1", "MacroAUPRC"]:
    metric_df = combined_df[metric].to_frame().T
    metric_df.index = ["CloudPred"]
    metric_df.to_csv(os.path.join(OUTPUT_BASE, f"{metric}.csv"))

print("\nSaved individual metric CSVs:")
print(os.path.join(OUTPUT_BASE, "AUROC.csv"))
print(os.path.join(OUTPUT_BASE, "MacroF1.csv"))
print(os.path.join(OUTPUT_BASE, "MacroAUPRC.csv"))
