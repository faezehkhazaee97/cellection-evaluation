# Batch mixing of cell-level representations

If CELLECTION is told which study each cell came from, do its per-cell representations mix better across studies while still separating cell types? Several human pancreas studies serve as the technical batches. This came up in Reviewer 1's question 4, on batch effects and subtype identification.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/pancreas_batch_mixing
python pancreas_embedding_with_batch.py      # instance-encoder embeddings, with batch covariate
python pancreas_embedding_without_batch.py   # instance-encoder embeddings, without batch covariate
python pancreas_attention_with_batch.py      # attention scores, with batch covariate (USE_BATCH = False for the no-batch arm)
```

All three scripts read `$DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad` and use its obs columns `dataset` (the study, used both as batch and as bag), `celltype` and `train_val`. They need the `cellection` package, scanpy, torch and scikit-learn, and use a GPU when one is available, otherwise the CPU. The embedding scripts make the full expression matrix dense, so give the job enough memory. Run times were not recorded; each script trains a single model.

## Files

- `pancreas_attention_with_batch.py`: trains CELLECTION with one bag per study and `train_val` as a placeholder label (seed 1), then scores the per-cell attention outputs by batch silhouette, cell-type silhouette, and KMeans ARI/NMI against cell type.
- `pancreas_embedding_with_batch.py`: trains the same model with the study as batch covariate and scores every cell's instance-encoder embedding with the same four metrics (seed 1).
- `pancreas_embedding_without_batch.py`: the same embedding evaluation with no batch covariate (seed 0).

## Outputs

Everything goes under `$RESULTS_DIR/R1-q4b/Results/`. Each CSV holds one row with `setting, ASW_batch, ASW_celltype, ARI, NMI`, where `ASW_batch` is the raw silhouette by batch, so lower means better mixing.
- `pancreas_batch_eval_with_batch.csv`: from `pancreas_attention_with_batch.py` (`pancreas_batch_eval_no_batch.csv` when `USE_BATCH = False`).
- `pancreas_batch_eval_celllevel_with_batch.csv`: from `pancreas_embedding_with_batch.py`.
- `pancreas_batch_eval_no_batch.csv`: from `pancreas_embedding_without_batch.py`. The attention script with `USE_BATCH = False` writes the same file name, so whichever runs last wins.
- `tmp_pancreas_batch_{baglevel,celllevel,nobatch}/train_val_classification_SEED<seed>/`: encoder and attention-model checkpoints.
