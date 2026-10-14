# Per-donor attribution in SEA-AD

Does a donor's predicted disease status follow where the model puts its attention, or just which glial cells the donor has? CELLECTION is trained on SEA-AD MTG glia, and for each held-out donor the predicted probability is correlated with the donor's share of attention and its share of cells in each glial supertype and major class (Figure S15). The probability throughout is for class 1, the second `disease` category, which in SEA-AD is `normal`; the AD probability is one minus it. This was Reviewer 2's question 3.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/seaad_donor_attribution
python donor_attribution.py
```

The only input is `$DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad`, with the obs columns `donor_id`, `disease` and `Supertype`. It uses a GPU when one is available and the CPU otherwise, and takes about 15 min on one GPU. Python packages: `cellection`, scanpy, torch, scipy, scikit-learn, matplotlib, seaborn.

## Files

- `donor_attribution.py`: splits donors 80/20 (seed 0), trains one model (AD vs control, one bag per donor), predicts the held-out donors, and correlates (Pearson and Spearman) their predicted class-1 probability with attention fraction and cell fraction per supertype and per major glial class.

## Outputs

All under `$RESULTS_DIR/R2-q3/Results/`:
- `donor_predictions.csv`: per held-out donor, `donor_id`, true label, predicted label and `prob` (class 1, `normal`).
- `attention_aggregated.csv`: attention fraction per donor and supertype, with the probability and true label.
- `celltype_attention_population_correlations.csv` and `celltype_attention_and_population_vs_prob.pdf`: correlations and scatter plots per supertype.
- `majorcell_attention_population_correlations.csv` and `majorcell_attention_and_population_vs_prob.pdf`: the same for microglia, oligodendrocytes, OPCs and astrocytes (Figure S15).

The model checkpoint goes to `./disease_classification_SEED<seed>/` in the folder the script is run from.
