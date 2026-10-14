#!/usr/bin/env python3
"""Supplementary hyper-parameter tables for the Figure 2 benchmark and for the MMIL baseline.

The table content is written out by hand in this file (TASKS and the *_rows() functions); nothing is parsed from
the benchmark scripts, so a change there has to be mirrored here. Writes, into folders that must already exist:
  $RESULTS_DIR/R2-q1/Code/Table_SX_Fig2_Hyperparameters_Comprehensive.csv
      one row per task and model variant (CELECTION, ProtoCell4P, DeepSet, CloudPred and the RF / logistic-
      regression variants of the three linear baselines): script, data and folds files, keys, split design,
      architecture or solver, training hyper-parameters, objective, model selection and notes
  $RESULTS_DIR/R2-q1/Code/benchmark_table_notes.md
      suggested table columns and code-level details worth stating in the supplement
  $RESULTS_DIR/R1-q5/Code/Table_SX_R1q5_MMIL_Comprehensive.csv
      the same description for the three MMIL tasks (mmil_baseline/)
Script paths in the tables are relative to benchmark_baselines/ (MMIL: mmil_baseline/).
"""
from __future__ import annotations

# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd

CODE_ROOT = Path(RESULTS_DIR) / "R2-q1" / "Code"
R1_CODE_ROOT = Path(RESULTS_DIR) / "R1-q5" / "Code"

FIG2_OUTPUT = CODE_ROOT / "Table_SX_Fig2_Hyperparameters_Comprehensive.csv"
MMIL_OUTPUT = R1_CODE_ROOT / "Table_SX_R1q5_MMIL_Comprehensive.csv"
NOTES_OUTPUT = CODE_ROOT / "benchmark_table_notes.md"

TASKS = {
    "Cardio": {
        "data_file": "Data/cardio/cardio_processed.h5ad",
        "folds_file": "Data/cardio/cardio_5folds.csv",
        "sample_key": "donor_id",
        "label_key": "disease",
        "celltype_key": "cell_type_leiden0.6",
        "class_num": 3,
        "input_representation": "Single-cell expression matrix grouped to donor bags.",
    },
    "CardioBiosample": {
        "data_file": "Data/cardio/cardio_processed.h5ad",
        "folds_file": "Data/cardio/cardio_biosample_id_5folds.csv",
        "sample_key": "biosample_id",
        "label_key": "disease",
        "celltype_key": "cell_type_leiden0.6",
        "class_num": 3,
        "input_representation": "Single-cell expression matrix grouped to biosample bags.",
    },
    "Covid": {
        "data_file": "Data/covid/covid_Ziegler_processed.h5ad",
        "folds_file": "Data/covid/covid_5folds.csv",
        "sample_key": "donor_id",
        "label_key": "disease_id",
        "celltype_key": "cell_type",
        "class_num": 2,
        "input_representation": (
            "Single-cell expression matrix grouped to donor bags; disease_id is derived in-script from "
            "disease__ontology_label with COVID-19/long COVID-19 as case and normal as control."
        ),
    },
    "Lupus": {
        "data_file": "Data/lupus/lupus_processed.h5ad",
        "folds_file": "Data/lupus/lupus_5folds.csv",
        "sample_key": "ind_cov",
        "label_key": "disease_cov",
        "celltype_key": "ct_cov",
        "class_num": 2,
        "input_representation": "Single-cell expression matrix grouped to individual bags.",
    },
    "Lupus_Race": {
        "data_file": "Data/lupus/lupus_processed.h5ad",
        "folds_file": "Data/lupus/race_5folds.csv",
        "sample_key": "ind_cov",
        "label_key": "pop_cov",
        "celltype_key": "ct_cov",
        "class_num": 2,
        "input_representation": "Single-cell expression matrix grouped to individual bags.",
    },
}


def fig2_common(task: str, model_family: str, model_variant: str, script_path: str) -> dict[str, object]:
    """Columns shared by every Figure 2 row: task description from TASKS plus model, script and CV design."""
    task_info = TASKS[task]
    return {
        "Task": task,
        "Model_Family": model_family,
        "Model_Variant": model_variant,
        "Script_Path": script_path,
        "Data_File": task_info["data_file"],
        "Folds_File": task_info["folds_file"],
        "Sample_Key": task_info["sample_key"],
        "Label_Key": task_info["label_key"],
        "Celltype_Key": task_info["celltype_key"],
        "Class_Num": task_info["class_num"],
        "Input_Representation": task_info["input_representation"],
        "CV_Design": "5-fold sample-level cross-validation",
        "Seed_Scheme": "10 random seeds (range(10))",
    }


def celection_rows() -> list[dict[str, object]]:
    script_map = {
        "Cardio": "celection/cardio.py",
        "CardioBiosample": "celection/cardio_biosample.py",
        "Covid": "celection/covid.py",
        "Lupus": "celection/lupus.py",
        "Lupus_Race": "celection/lupus_race.py",
    }
    rows = []
    for task, script_path in script_map.items():
        row = fig2_common(task, "CELECTION", "CELECTION", script_path)
        row["Folds_File"] = "Data/lupus/lupus_5folds.csv" if task == "Lupus_Race" else row["Folds_File"]
        row["Train_Val_Design"] = "Within each trainval fold, random 80/20 bag split via torch random_split"
        row["Architecture_or_Solver"] = (
            "CellEncoder FCNN [input->256->32] with LayerNorm, ReLU, dropout=0.1 and an extra 32->32 linear layer; "
            "PointNet backbone with 2 T-nets (32x32 input transform and 64x64 feature transform), SharedMLP(32->64) "
            "and SharedMLP(64->128), gated-attention pooling (attention hidden dim=32), classifier 128->64->k."
        )
        row["Training_Hyperparameters"] = (
            "Adam over encoder+classifier parameters, lr=1e-4, batch_size=5 bags, max_epochs=200, early_stopping_patience=20."
        )
        row["Loss_or_Objective"] = "CrossEntropyLoss applied to Softmax-normalized logits."
        row["Model_Selection"] = "Lowest validation loss checkpoint saved separately for encoder and classifier."
        row["Notable_Notes"] = (
            "bioPointNet_Apr2025 backbone. Lupus_Race script uses lupus_5folds.csv rather than race_5folds.csv."
            if task == "Lupus_Race"
            else "bioPointNet_Apr2025 backbone."
        )
        rows.append(row)
    return rows


def protocell_rows() -> list[dict[str, object]]:
    """Figure 2 table rows for ProtoCell4P, one per task (with its prototype count)."""
    task_to_proto = {
        "Cardio": ("protocell4p/cardio.py", 16),
        "CardioBiosample": ("protocell4p/cardio_biosample.py", 16),
        "Covid": ("protocell4p/covid.py", 16),
        "Lupus": ("protocell4p/lupus.py", 8),
        "Lupus_Race": ("protocell4p/lupus_race.py", 8),
    }
    rows = []
    for task, (script_path, n_proto) in task_to_proto.items():
        row = fig2_common(task, "ProtoCell4P", f"proto_{n_proto}", script_path)
        row["Train_Val_Design"] = "Within each trainval fold, sample-level 80/20 split via sklearn train_test_split(random_state=SEED)"
        row["Architecture_or_Solver"] = (
            f"ProtoCell with encoder input->128->128->32, decoder 32->128->128->input, importance network "
            f"input->128->128->({n_proto} x class_num), learned prototype matrix {n_proto}x32, auxiliary cell-type "
            "heads enabled when cell-type labels are present."
        )
        row["Training_Hyperparameters"] = (
            f"Adam, lr=1e-4, batch_size=4, max_epoch=20, test_step=1, h_dim=128, z_dim=32, n_layers=2, "
            f"n_proto={n_proto}, d_min=1, pretraining disabled (max_epoch_pretrain=0)."
        )
        row["Loss_or_Objective"] = (
            "Weighted sum of reconstruction, cell-to-prototype, prototype-to-cell, prototype-separation, "
            "cell-type auxiliary, and classification auxiliary losses; lambda_1..lambda_6 all set to 1 "
            "(lambda_5/lambda_6 would be zeroed only if n_ct were absent)."
        )
        row["Model_Selection"] = "Best validation checkpoint retained at test_step intervals."
        row["Notable_Notes"] = "Uses task-specific prototype count (16 for cardio/covid tasks, 8 for lupus tasks)."
        rows.append(row)
    return rows


def deepset_rows() -> list[dict[str, object]]:
    script_map = {
        "Cardio": "deepset/cardio.py",
        "CardioBiosample": "deepset/cardio_biosample.py",
        "Covid": "deepset/covid.py",
        "Lupus": "deepset/lupus.py",
        "Lupus_Race": "deepset/lupus_race.py",
    }
    rows = []
    for task, script_path in script_map.items():
        row = fig2_common(task, "DeepSet", "DeepSet", script_path)
        row["Train_Val_Design"] = (
            "Within each trainval fold, sample-level 80/20 split via sklearn train_test_split(random_state=SEED)."
        )
        row["Architecture_or_Solver"] = (
            "Per-cell MLP Linear(input,5)->ReLU->Linear(5,5)->ReLU, cloudpred.utils.Aggregator pooling layer, "
            "bag-level MLP Linear(5,5)->ReLU->Linear(5,k)."
        )
        row["Training_Hyperparameters"] = (
            "cloudpred.utils.train_classifier with centers=5, iterations=100, eta=1e-4, stochastic=True, regularize=None; "
            "outer scripts also use iterations=25 during sample packaging."
        )
        row["Loss_or_Objective"] = "Classifier training objective inside cloudpred.utils.train_classifier."
        row["Model_Selection"] = "Best internal validation loss as returned by cloudpred.utils.train_classifier."
        row["Notable_Notes"] = "Scripts also emit a separate PC_ result set, but the main DeepSet rows remain the benchmark entry."
        rows.append(row)
    return rows


def cloudpred_rows() -> list[dict[str, object]]:
    script_map = {
        "Cardio": "cloudpred/cardio.py",
        "CardioBiosample": "cloudpred/cardio_biosample.py",
        "Covid": "cloudpred/covid.py",
        "Lupus": "cloudpred/lupus.py",
        "Lupus_Race": "cloudpred/lupus_race.py",
    }
    rows = []
    for task, script_path in script_map.items():
        row = fig2_common(task, "CloudPred", "CloudPred", script_path)
        row["Train_Val_Design"] = (
            "Within each trainval fold, sample-level 80/20 split via sklearn train_test_split(random_state=SEED)."
        )
        row["Architecture_or_Solver"] = (
            "GaussianMixture(n_components=5, covariance_type='diag') fit on pooled training cells, followed by "
            "cloudpred DensityClassifier over mixture memberships."
        )
        row["Training_Hyperparameters"] = (
            "For each lr in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3], run SGD(momentum=0.9) for 1000 iterations; "
            "outer scripts also use iterations=25 during sample packaging."
        )
        row["Loss_or_Objective"] = "CrossEntropyLoss on DensityClassifier outputs."
        row["Model_Selection"] = "Best validation loss tracked across iterations and learning-rate grid."
        row["Notable_Notes"] = "Scripts also emit a separate PC_ result set, but the main CloudPred rows remain the benchmark entry."
        rows.append(row)
    return rows


def baseline_row(
    task: str,
    family: str,
    variant: str,
    script_path: str,
    input_representation: str,
    notes: str = "",
) -> dict[str, object]:
    """One Figure 2 row for a linear baseline; variant is "RF" or "LogisticRegression"."""
    row = fig2_common(task, family, variant, script_path)
    row["Train_Val_Design"] = "Within each trainval fold, sample-level 80/20 split via sklearn train_test_split(random_state=SEED)."
    row["Architecture_or_Solver"] = input_representation
    if variant == "RF":
        row["Training_Hyperparameters"] = "RandomForestClassifier(random_state=SEED) with sklearn defaults."
        row["Loss_or_Objective"] = "Default sklearn RandomForestClassifier objective."
    else:
        row["Training_Hyperparameters"] = "OneVsRestClassifier(LogisticRegression(random_state=SEED)) with sklearn defaults."
        row["Loss_or_Objective"] = "Default sklearn LogisticRegression objective inside one-vs-rest wrapper."
    row["Model_Selection"] = "Fit once per train/val split; no separate hyperparameter search."
    row["Notable_Notes"] = notes
    return row


def baseline_rows() -> list[dict[str, object]]:
    """Figure 2 rows for the CT pseudobulk, CT proportion and pseudobulk baselines (RF and logistic regression)."""
    rows: list[dict[str, object]] = []

    ct_pb_scripts = {
        "Cardio": ("ct_pseudobulk/cardio.py", True, True),
        "CardioBiosample": ("ct_pseudobulk/cardio_biosample.py", False, True),
        "Covid": ("ct_pseudobulk/covid.py", True, True),
        "Lupus": ("ct_pseudobulk/lupus.py", True, True),
        "Lupus_Race": ("ct_pseudobulk/lupus_race.py", True, True),
    }
    for task, (script_path, has_rf, has_logreg) in ct_pb_scripts.items():
        input_repr = "Cell-type-specific pseudobulk vectors concatenated across cell types; per-sample library-size normalized where implemented."
        if has_rf:
            rows.append(baseline_row(task, "CT_pseudobulk", "RF", script_path, input_repr))
        if has_logreg:
            note = "CardioBiosample script only executes LogisticRegression; no RandomForest model is fit (the RF result dict stays empty)." if task == "CardioBiosample" else ""
            rows.append(baseline_row(task, "CT_pseudobulk", "LogisticRegression", script_path, input_repr, note))

    ct_prop_scripts = {
        "Cardio": "ct_proportion/cardio.py",
        "CardioBiosample": "ct_proportion/cardio_biosample.py",
        "Covid": "ct_proportion/covid.py",
        "Lupus": "ct_proportion/lupus.py",
        "Lupus_Race": "ct_proportion/lupus_race.py",
    }
    for task, script_path in ct_prop_scripts.items():
        input_repr = "Per-sample cell-type proportion vector."
        rows.append(baseline_row(task, "CT_proportion", "RF", script_path, input_repr))
        rows.append(baseline_row(task, "CT_proportion", "LogisticRegression", script_path, input_repr))

    pb_scripts = {
        "Cardio": ("pseudobulk/cardio.py", True, True),
        "CardioBiosample": ("pseudobulk/cardio_biosample.py", False, True),
        "Covid": ("pseudobulk/covid.py", True, True),
        "Lupus": ("pseudobulk/lupus.py", True, True),
        "Lupus_Race": ("pseudobulk/lupus_race.py", True, True),
    }
    for task, (script_path, has_rf, has_logreg) in pb_scripts.items():
        input_repr = "Per-sample pseudobulk gene-expression vector computed as mean expression; library-size normalized where implemented."
        if has_rf:
            rows.append(baseline_row(task, "Pseudobulk", "RF", script_path, input_repr))
        if has_logreg:
            note = "CardioBiosample script only executes LogisticRegression; RF result containers remain empty placeholders." if task == "CardioBiosample" else ""
            rows.append(baseline_row(task, "Pseudobulk", "LogisticRegression", script_path, input_repr, note))

    return rows


def mmil_rows() -> list[dict[str, object]]:
    return [
        {
            "Task": "Covid",
            "Model_Family": "MMIL",
            "Model_Variant": "Ridge-logistic MMIL",
            "Script_Path": "mmil_covid.R",
            "Prep_or_Data_File": "Data/covid/covid_Ziegler_processed.h5ad -> Results/Covid/cells.csv",
            "Folds_File": "Data/covid/covid_5folds.csv",
            "Sample_Key": "donor_id",
            "Label_Key": "bag_label/case_control/disease__ontology_label-derived binary case-control",
            "Class_Num": 2,
            "Input_Representation": "Cell-level gene expression exported to cells.csv and converted to dgCMatrix inside R.",
            "CV_Design": "5-fold donor-level cross-validation",
            "Train_Val_Design": "No separate validation split; cv.glmnet performs internal lambda selection on the training fold.",
            "Seed_Scheme": "No outer seed loop in the MMIL R scripts.",
            "Architecture_or_Solver": "cv.glmnet with alpha=0 (ridge) and family='binomial'.",
            "Training_Hyperparameters": "Thresholded at 0.5 for class labels; one model fit per outer fold.",
            "Loss_or_Objective": "Penalized binomial likelihood with lambda selected at lambda.min.",
            "Model_Selection": "Internal cv.glmnet selection of lambda.min.",
            "Notable_Notes": "No cell subsampling in the COVID prep script.",
        },
        {
            "Task": "Lupus",
            "Model_Family": "MMIL",
            "Model_Variant": "Ridge-logistic MMIL",
            "Script_Path": "mmil_lupus_disease.R",
            "Prep_or_Data_File": "Data/lupus/lupus_processed.h5ad -> Results/Lupus_Disease/cells.csv",
            "Folds_File": "Data/lupus/lupus_5folds.csv",
            "Sample_Key": "ind_cov",
            "Label_Key": "disease_cov == 'sle'",
            "Class_Num": 2,
            "Input_Representation": "Cell-level gene expression exported to cells.csv and converted to dgCMatrix inside R.",
            "CV_Design": "5-fold individual-level cross-validation",
            "Train_Val_Design": "No separate validation split; cv.glmnet performs internal lambda selection on the training fold.",
            "Seed_Scheme": "No outer seed loop in the MMIL R scripts.",
            "Architecture_or_Solver": "cv.glmnet with alpha=0 (ridge) and family='binomial'.",
            "Training_Hyperparameters": "Prep script downsamples each donor to 10% of cells before writing cells.csv.",
            "Loss_or_Objective": "Penalized binomial likelihood with lambda selected at lambda.min.",
            "Model_Selection": "Internal cv.glmnet selection of lambda.min.",
            "Notable_Notes": "Binary labels are validated to be donor-consistent before training.",
        },
        {
            "Task": "Lupus_Race",
            "Model_Family": "MMIL",
            "Model_Variant": "Ridge-logistic MMIL",
            "Script_Path": "mmil_lupus_race.R",
            "Prep_or_Data_File": "Data/lupus/lupus_processed.h5ad -> Results/Lupus_Race/cells.csv",
            "Folds_File": "Data/lupus/race_5folds.csv",
            "Sample_Key": "ind_cov",
            "Label_Key": "pop_cov == 'asian'",
            "Class_Num": 2,
            "Input_Representation": "Cell-level gene expression exported to cells.csv and converted to dgCMatrix inside R.",
            "CV_Design": "5-fold individual-level cross-validation",
            "Train_Val_Design": "No separate validation split; cv.glmnet performs internal lambda selection on the training fold.",
            "Seed_Scheme": "No outer seed loop in the MMIL R scripts.",
            "Architecture_or_Solver": "cv.glmnet with alpha=0 (ridge) and family='binomial'.",
            "Training_Hyperparameters": "Prep script downsamples each donor to 10% of cells before writing cells.csv.",
            "Loss_or_Objective": "Penalized binomial likelihood with lambda selected at lambda.min.",
            "Model_Selection": "Internal cv.glmnet selection of lambda.min.",
            "Notable_Notes": "Positive class is hard-coded to asian after lower-casing the population labels.",
        },
    ]


def build_notes() -> str:
    """Markdown text of benchmark_table_notes.md."""
    return """# Benchmark Table Additions

Recommended supplementary-table columns to add:

- `Script_Path`
- `Data_File`
- `Folds_File`
- `Sample_Key`
- `Label_Key`
- `Celltype_Key`
- `Input_Representation`
- `Train_Val_Design`
- `Architecture_or_Solver`
- `Training_Hyperparameters`
- `Loss_or_Objective`
- `Model_Selection`
- `Notable_Notes`

Code-level details worth documenting explicitly:

- CELECTION uses a PointNet-style backbone with **two T-nets** plus gated-attention pooling.
- ProtoCell4P uses task-specific prototype counts: 16 for Cardio/CardioBiosample/Covid and 8 for Lupus/Lupus_Race.
- CELECTION `Lupus_Race` currently reads `lupus_5folds.csv`, while the other Lupus_Race benchmark scripts use `race_5folds.csv`.
- `ct_pseudobulk/cardio_biosample.py` only runs LogisticRegression; no RandomForest model is fit.
- `pseudobulk/cardio_biosample.py` only runs LogisticRegression; RF result containers exist but are not populated.
- MMIL has no outer seed loop and no separate validation split; `cv.glmnet` performs the inner lambda selection.
"""


def main() -> None:
    fig2_rows = celection_rows() + protocell_rows() + deepset_rows() + cloudpred_rows() + baseline_rows()
    pd.DataFrame(fig2_rows).to_csv(FIG2_OUTPUT, index=False)

    pd.DataFrame(mmil_rows()).to_csv(MMIL_OUTPUT, index=False)
    NOTES_OUTPUT.write_text(build_notes())

    print(f"Wrote {FIG2_OUTPUT}")
    print(f"Wrote {MMIL_OUTPUT}")
    print(f"Wrote {NOTES_OUTPUT}")


if __name__ == "__main__":
    main()
