"""Shared inputs of the Parse-PBMC benchmark (written to benchmark.parse_dir).

deg_mapped.csv              published Parse 10M PBMC DEGs; dictionary cell types mapped to the 18 cell
                            types of the h5ad (coarse labels Granulocyte, B_cell, CD4_T_cell,
                            CD8_T_cell, NK, Mono dropped)
background_by_celltype.csv  genes detected (count > 0) in >= 5% of the cells of each cell type
benchmark/responsive_conditions.csv  (cell type, cytokine) pairs with >= 10 DEGs (adj. p < 0.05, |log FC| >= 0.25):
                            the common set of conditions scored for every method
SecAct_input/pseudobulk_log2fc.csv  SecAct / CytoSig input: one pseudobulk log fold change per responsive condition,
                            as in SecAct.CCC.scRNAseq (step 2): counts summed over the cells of the cell type in the
                            condition -> CPM -> log2(x + 1), cytokine minus PBS of the same cell type; conditions with
                            < 30 cells in either group are skipped (as in SecAct; SecAct_input/skipped_conditions.csv)

python 09_benchmark_parse/prepare_inputs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from paper_utils import bench, path

LOGFC_CUT, PADJ_CUT, MIN_DEG, MIN_FRAC, MIN_CELLS = 0.25, 0.05, 10, 0.05, 30
DROP = {"Granulocyte", "B_cell", "CD4_T_cell", "CD8_T_cell", "NK", "Mono"}
DEG_TO_ADATA = {
    "CD4_Memory_T_cell": "CD4 Memory", "CD4_Naive_T_cell": "CD4 Naive",
    "CD8_Memory_T_cell": "CD8 Memory", "CD8_Naive_T_cell": "CD8 Naive",
    "CD14_Mono": "CD14 Mono", "CD16_Mono": "CD16 Mono",
    "Naive_B_cell": "B Naive", "Intermediate_B_cell": "B Intermediate/Memory",
    "NK_CD56hi": "NK CD56bright", "NK_CD56low": "NK",
    "Treg": "Treg", "NKT": "NKT", "MAIT": "MAIT", "ILC": "ILC",
    "HSPC": "HSPC", "pDC": "pDC", "cDC": "cDC", "Plasmablast": "Plasmablast",
}
CHUNK = 500_000

A = ad.read_h5ad(path("parse", "counts"), backed="r")  # raw counts
celltypes = A.obs["cell_type"].astype(str).to_numpy()

# ---------------------------------------------------------------- DEGs
deg = pd.read_csv(path("parse", "degs"))
deg = deg[~deg["celltype"].isin(DROP)].copy()
deg["celltype_adata"] = deg["celltype"].map(DEG_TO_ADATA)
assert deg["celltype_adata"].notna().all(), "unmapped DEG cell types"
assert set(deg["celltype_adata"]) <= set(celltypes), "DEG cell types missing from the h5ad"
deg[["gene", "log_fc", "adj_p_value", "celltype", "celltype_adata", "cytokine"]].to_csv(
    bench("parse_dir", "deg_mapped.csv", new=True), index=False)

# ---------------------------------------------------------------- responsive conditions
sig = deg[(deg["cytokine"] != "PBS") & (deg["adj_p_value"] < PADJ_CUT) & (deg["log_fc"].abs() >= LOGFC_CUT)]
resp = (sig.groupby(["celltype_adata", "cytokine"]).size().rename("n_deg").reset_index()
        .query("n_deg >= @MIN_DEG").rename(columns={"celltype_adata": "celltype"})
        .sort_values(["celltype", "cytokine"]).reset_index(drop=True))
resp.to_csv(bench("parse_dir", "benchmark", "responsive_conditions.csv", new=True), index=False)
print(f"responsive conditions: {len(resp)}")

# ---------------------------------------------------------------- background genes
cats = pd.Categorical(celltypes)
codes, names = cats.codes, cats.categories
detected = np.zeros((len(names), A.n_vars), dtype=np.int64)
for s in range(0, A.n_obs, CHUNK):
    X = A[s:s + CHUNK].to_memory().X.tocsr()
    X.data = (X.data > 0).astype(np.int64)
    for k in np.unique(codes[s:s + CHUNK]):
        detected[k] += np.asarray(X[codes[s:s + CHUNK] == k].sum(0)).ravel()
n = np.bincount(codes, minlength=len(names))
rows = [{"celltype": ct, "gene": g} for k, ct in enumerate(names)
        for g in A.var_names[detected[k] / n[k] >= MIN_FRAC]]
pd.DataFrame(rows).to_csv(bench("parse_dir", "background_by_celltype.csv", new=True), index=False)
print(f"background genes per cell type:\n{pd.DataFrame(rows)['celltype'].value_counts().to_string()}")

# ---------------------------------------------------------------- SecAct / CytoSig input
def pseudobulk_log2cpm(mask):
    idx = np.where(mask)[0]
    if len(idx) < MIN_CELLS:
        return None
    X = A[idx].to_memory().X
    tot = np.asarray((X.tocsr() if issparse(X) else X).sum(0)).ravel()
    return np.log2(tot * 1e6 / tot.sum() + 1)


cyt_arr = A.obs["cytokine"].astype(str).to_numpy()
pbs = {c: pseudobulk_log2cpm((celltypes == c) & (cyt_arr == "PBS")) for c in resp["celltype"].unique()}
cols, skipped = {}, []
for r in resp.itertuples():
    stim, ctrl = pseudobulk_log2cpm((celltypes == r.celltype) & (cyt_arr == r.cytokine)), pbs[r.celltype]
    if stim is None or ctrl is None:
        skipped.append((r.celltype, r.cytokine))
        continue
    cols[f"{r.celltype}|{r.cytokine}"] = stim - ctrl

pd.DataFrame(cols, index=A.var_names).to_csv(bench("parse_dir", "SecAct_input", "pseudobulk_log2fc.csv", new=True))
pd.DataFrame(skipped, columns=["celltype", "cytokine"]).to_csv(
    bench("parse_dir", "SecAct_input", "skipped_conditions.csv"), index=False)
print(f"conditions: {len(cols)} | skipped (< {MIN_CELLS} cells): {len(skipped)}")
