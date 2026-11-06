"""IREA input for the in vivo datasets ('Gene Matrix' mode of https://www.immune-dictionary.org/irea/).

Per cell type: mean log-normalized expression per condition (conditions with >= 10 cells), genes with
mean > 0.25 in at least one condition, treated minus control, each column scaled to [-1, 1].

  G-CSF bone marrow, Mye_lineage (post vs pre)     -> IREA/GCSF_BM_Mye_lineage_delta.xlsx
  IL-7 T cells, CD4_T and CD8_T (post_dose1_wk1 vs baseline) -> IREA/IL7_CD4_T_delta.xlsx, IL7_CD8_T_delta.xlsx

Then, on the IREA website: Gene Matrix, species Human, cell type Monocyte (G-CSF; proxy for the
myeloid lineage), T_cell_CD4 / T_cell_CD8 (IL-7). Save the result tables as
IREA/IREA_BM_Mye_Monocyte.csv, IREA/IREA_combined_CD4T.csv, IREA/IREA_combined_CD8T.csv.

python 10_benchmark_in_vivo/irea_input.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from paper_utils import bench, path


def build_irea_input(adata, condition_key, control, celltype_key, celltypes, prefix, min_mean=0.25, min_cells=10):
    X = adata.X.tocsr() if sparse.issparse(adata.X) else np.asarray(adata.X)
    if X.max() > 50:
        raise ValueError("looks like raw counts: log-normalized expression is required")
    genes = pd.Index(adata.var_names.astype(str))
    cts, conds = adata.obs[celltype_key].astype(str), adata.obs[condition_key].astype(str)
    for ct in celltypes:
        m_ct = (cts == ct).to_numpy()
        means = {}
        for cond in sorted(conds[m_ct].unique()):
            m = m_ct & (conds == cond).to_numpy()
            if m.sum() >= min_cells:
                means[cond] = np.asarray(X[m].mean(axis=0)).ravel()
        if control not in means or len(means) < 2:
            print(f"[skip] {ct}: control or treated condition missing")
            continue
        df = pd.DataFrame(means, index=genes)
        df = df.loc[df.max(axis=1) > min_mean]
        delta = pd.DataFrame({f"{c}_to_{control}": df[c] - df[control] for c in means if c != control}, index=df.index)
        peak = delta.abs().max()
        peak[peak == 0] = 1.0
        delta = delta / peak
        delta.index.name = None
        f = bench("in_vivo_dir", "IREA", f"{prefix}_{re.sub(r'[^0-9A-Za-z._-]+', '_', ct)}_delta.xlsx", new=True)
        delta.to_excel(f)
        print(f"{ct}: {delta.shape} -> {f}")


gcsf = sc.read_h5ad(path("inference", "gcsf"))
build_irea_input(gcsf, "condition", "pre", "celltype", ["Mye_lineage"], "GCSF_BM")
il7 = sc.read_h5ad(path("inference", "il7"))
build_irea_input(il7, "treatment", "baseline", "celltype", ["CD4_T", "CD8_T"], "IL7")
