"""huCIRA on the Parse-PBMC benchmark (huCIRA environment).

Per responsive condition: hucira.run_one_enrichment_test on the log-normalized Parse data, contrast
cytokine vs PBS in the Parse cell type, gene sets of the matching Human Cytokine Dictionary cell type,
up-regulated direction, default settings of the huCIRA authors (|log FC| >= 1, p < 0.01, gene-set size
10-1000, 1,000 permutations). Cytokines ranked by NES; true_rank = rank of the cytokine;
pool_size = number of cytokines ranked in that condition.
Note: the dictionary is derived from the same Parse data, so huCIRA is not an independent comparator.

Output: huCIRA_output/hucira_top10_results_default.csv

python 09_benchmark_parse/run_hucira.py [--threads 50]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hucira as hc
import numpy as np
import pandas as pd
import scanpy as sc

from paper_utils import bench, path

HUCIRA_CT = {  # Parse cell type -> huCIRA dictionary cell type
    "CD4 Memory": "CD4_Memory_T_cell", "CD4 Naive": "CD4_Naive_T_cell", "CD8 Memory": "CD8_Memory_T_cell",
    "CD8 Naive": "CD8_Naive_T_cell", "CD14 Mono": "CD14_Mono", "CD16 Mono": "CD16_Mono",
    "B Naive": "Naive_B_cell", "B Intermediate/Memory": "Intermediate_B_cell",
    "NK CD56bright": "NK_CD56hi", "NK": "NK_CD56low", "Treg": "Treg", "NKT": "NKT", "MAIT": "MAIT",
    "ILC": "ILC", "HSPC": "HSPC", "pDC": "pDC", "cDC": "cDC", "Plasmablast": "Plasmablast",
}
DEFAULT = dict(threshold_lfc=1.0, threshold_pval=0.01, min_size=10, max_size=1000,
               threshold_expression=0.0, permutation_num=1000)

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=50)
a = p.parse_args()
out = bench("parse_dir", "huCIRA_output", "hucira_top10_results_default.csv", new=True)

adata = sc.read_h5ad(path("parse", "normalized"))
hcd = hc.load_human_cytokine_dict(save_dir=str(path("reference", "hucira_dir")), force_download=False)
responsive = pd.read_csv(path("benchmark", "parse_dir") / "benchmark" / "responsive_conditions.csv")


def score(parse_ct, cyt, ctrl="PBS", top_n=10):
    base = dict(celltype=parse_ct, cytokine=cyt, true_rank=np.nan, hit=False)
    hct = HUCIRA_CT.get(parse_ct)
    if hct is None:
        return {**base, "status": "no_celltype_map", "pool_size": np.nan}
    try:
        er = hc.run_one_enrichment_test(adata=adata, df=hcd, contrast_combo=(cyt, ctrl),
                                        celltype_combo=(parse_ct, hct), contrast_column="cytokine",
                                        celltype_column="cell_type", direction="upregulated",
                                        threads=a.threads, **DEFAULT)
    except Exception as e:
        return {**base, "status": f"error:{type(e).__name__}", "pool_size": np.nan}
    if er is None or len(er) == 0 or cyt not in set(er["cytokine"]):
        return {**base, "status": "no_result", "pool_size": int(len(er)) if er is not None else 0}
    ranking = er.sort_values("NES", ascending=False).reset_index(drop=True)
    rank = int(ranking.index[ranking["cytokine"] == cyt][0]) + 1
    return dict(celltype=parse_ct, cytokine=cyt, true_rank=rank, hit=bool(rank <= top_n),
                status="scored", pool_size=int(len(er)))


res = pd.DataFrame([score(r.celltype, r.cytokine) for r in responsive.itertuples()])
res.to_csv(out, index=False)
print(res["status"].value_counts().to_string())
print(f"top-10 recall (all conditions): {(res['true_rank'] <= 10).fillna(False).mean():.3f}")
