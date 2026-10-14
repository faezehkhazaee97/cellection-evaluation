"""GSE132044 (Ding et al., Nat Biotechnol 2020) as an h5ad, for the PBMC simulations.

GSE132044 profiles PBMCs with several scRNA-seq protocols. Download and gunzip these files into
$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/ and run the script from there; it reads and writes in the
current directory:
    GSE132044_pbmc_hg38_count_matrix.mtx  counts, genes x cells
    GSE132044_pbmc_hg38_gene.tsv          gene names, "<ENSEMBL>_<SYMBOL>"
    GSE132044_pbmc_hg38_cell.tsv          cell names, e.g. "PBMC1.Smart-seq2.p2_A1" (donor, protocol)
Output: pbmc_smartseq_raw.h5ad, raw counts, cells x genes. No cell is filtered by protocol, so
the "smartseq" batch in the simulations is the whole matrix, not only its Smart-seq2 cells.
"""
import scanpy as sc
import pandas as pd

# the .mtx is genes x cells
adata = sc.read_mtx("GSE132044_pbmc_hg38_count_matrix.mtx").T

genes = pd.read_csv("GSE132044_pbmc_hg38_gene.tsv", sep="\t", header=None)
cells = pd.read_csv("GSE132044_pbmc_hg38_cell.tsv", sep="\t", header=None)

adata.var_names = genes[0].values
adata.obs_names = cells[0].values

adata.write_h5ad("pbmc_smartseq_raw.h5ad")