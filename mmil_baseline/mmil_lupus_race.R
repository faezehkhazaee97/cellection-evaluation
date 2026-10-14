# MMIL baseline, lupus ancestry task (Asian vs White). The model code is identical in mmil_covid.R
# and mmil_lupus_disease.R; only the usage message differs.
# Run as: Rscript mmil_lupus_race.R <cells.csv> <folds.csv> <output_dir>
#   cells.csv: one row per cell (cell_id, donor_id, gene columns, bag_label), from
#              prep_lupus_race.py or runner/run_mmil.py
#   folds.csv: donor_id -> fold (other columns are ignored)
# Every cell inherits its donor's bag_label. For each fold, a ridge logistic regression (cv.glmnet,
# alpha = 0, lambda.min) is fit on the cells of the other folds' donors and scores every cell of the
# held-out donors; AUROC, F1 (cut-off 0.5) and macro AUPRC are computed over those cells.
# Writes fold_<k>_preds.csv (cell_id, donor_id, label, prob) and MMIL_summary.csv (fold, AUROC, F1,
# MacroAUPRC). A fold with one class in train or test is skipped and gets no row.
# R packages: Matrix, glmnet, pROC, dplyr, caret.
suppressPackageStartupMessages({
    library(Matrix)
    library(glmnet)
    library(pROC)
    library(dplyr)
    library(caret)     # Required for F_meas()
})
options(stringsAsFactors = FALSE)
# Step-wise PR-AUC (average-precision style): rank cells by score, sum recall increments x precision.
pr_auc <- function(y_true, y_score) {
    if (sum(y_true == 1) == 0) {
        return(NA_real_)
    }
    ord <- order(y_score, decreasing = TRUE)
    y <- y_true[ord]
    tp <- cumsum(y == 1)
    fp <- cumsum(y == 0)
    recall <- tp / sum(y == 1)
    precision <- tp / (tp + fp)
    recall <- c(0, recall)
    precision <- c(1, precision)
    auc <- sum((recall[-1] - recall[-length(recall)]) * precision[-1])
    return(auc)
}
# Macro AUPRC: mean of the class-1 PR-AUC and the class-0 PR-AUC (scores flipped to 1 - p).
compute_macro_auprc <- function(y_true, y_score) {
    pos <- pr_auc(y_true, y_score)
    neg <- pr_auc(1 - y_true, 1 - y_score)
    mean(c(pos, neg), na.rm = TRUE)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript mmil_lupus_race.R <cells.csv> <folds.csv> <output_dir>")
}
cells_path <- args[1]     # /path/to/cells.csv
folds_path <- args[2]     # /path/to/folds.csv
out_path   <- args[3]     # /path/to/output/
dir.create(out_path, showWarnings = FALSE, recursive = TRUE)
cat("Loading data...\n")
cells <- read.csv(cells_path)
folds <- read.csv(folds_path)
set.seed(0)  # fixes cv.glmnet's internal lambda-selection folds
cat("Merging folds...\n")
# Inner join: cells whose donor is not in the fold file are dropped.
cells <- merge(cells, folds, by = "donor_id")
feature_cols <- setdiff(
    colnames(cells),
    c("cell_id", "donor_id", "bag_label", "fold")
)
cat("Creating sparse matrix...\n")
feature_df <- cells[, feature_cols, drop = FALSE]
numeric_cols <- sapply(feature_df, is.numeric)
if (!all(numeric_cols)) {
    bad_cols <- names(feature_df)[!numeric_cols]
    message("Dropping non-numeric feature columns: ",
            paste(bad_cols, collapse = ", "))
    feature_df <- feature_df[, numeric_cols, drop = FALSE]
}
X_all <- as(as.matrix(feature_df), "dgCMatrix")
y_all <- cells$bag_label
fold_ids <- cells$fold
donor_ids <- cells$donor_id
auroc_vec <- c()
f1_vec <- c()
auprc_vec <- c()
res_folds <- c()
for (f in sort(unique(fold_ids))) {
    cat("\n==============================\n")
    cat(sprintf("Running Fold %d\n", f))
    cat("==============================\n")
    test_donors <- folds$donor_id[folds$fold == f]
    test_idx  <- which(donor_ids %in% test_donors)
    train_idx <- which(!(donor_ids %in% test_donors))
    cat(sprintf("Train cells: %d | Test cells: %d\n",
                length(train_idx), length(test_idx)))
    X_train <- X_all[train_idx, ]
    y_train <- y_all[train_idx]
    X_test <- X_all[test_idx, ]
    y_test <- y_all[test_idx]
    # glmnet wants numeric 0/1 labels
    y_train_num <- as.numeric(as.character(y_train))
    y_test_num  <- as.numeric(as.character(y_test))
    if (length(unique(y_train_num)) < 2 || length(unique(y_test_num)) < 2) {
        message("Skipping fold ", f, ": only one class in train or test.")
        next
    }
    fit <- cv.glmnet(
        x = X_train,
        y = y_train_num,
        alpha = 0,          # Ridge regression
        family = "binomial"
    )
    preds <- predict(fit, newx = X_test, type = "response", s = "lambda.min")
    preds <- as.numeric(preds)
    pred_labels <- ifelse(preds >= 0.5, 1, 0)
    auroc <- auc(y_test_num, preds)
    macro_auprc <- compute_macro_auprc(y_test_num, preds)
    f1 <- caret::F_meas(
        factor(pred_labels, levels = c(0,1)),
        factor(y_test_num, levels = c(0,1))
    )
    auroc_vec <- c(auroc_vec, auroc)
    f1_vec    <- c(f1_vec, f1)
    auprc_vec <- c(auprc_vec, macro_auprc)
    res_folds <- c(res_folds, f)
    fold_res <- data.frame(
        cell_id  = cells$cell_id[test_idx],
        donor_id = donor_ids[test_idx],
        label    = y_test_num,
        prob     = preds
    )
    write.csv(
        fold_res,
        file = sprintf("%s/fold_%d_preds.csv", out_path, f),
        row.names = FALSE
    )
    cat(sprintf("Fold %d AUROC = %.4f | F1 = %.4f | MacroAUPRC = %.4f\n",
                f, auroc, f1, macro_auprc))
}
cat("\n===== FINAL RESULTS =====\n")
cat(sprintf("Mean AUROC: %.4f\n", mean(auroc_vec)))
cat(sprintf("Mean F1: %.4f\n", mean(f1_vec)))
cat(sprintf("Mean MacroAUPRC: %.4f\n", mean(auprc_vec)))
summary_df <- data.frame(
    fold = res_folds,
    AUROC = auroc_vec,
    F1 = f1_vec,
    MacroAUPRC = auprc_vec
)
write.csv(
    summary_df,
    file = sprintf("%s/MMIL_summary.csv", out_path),
    row.names = FALSE
)
