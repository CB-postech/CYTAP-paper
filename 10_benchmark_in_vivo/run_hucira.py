"""huCIRA on the in vivo datasets (huCIRA environment), default settings of the huCIRA authors
(|log FC| >= 1, p < 0.01, gene-set size 10-1000, 1,000 permutations), up-regulated direction.

  G-CSF bone marrow, Mye_lineage post vs pre; dictionary cell type Granulocyte as proxy (the dictionary
      has no myeloid-lineage cell type)
  IL-7 T cells, CD4_T -> CD4_T_cell, CD8_T -> CD8_T_cell, post_dose1_wk1 vs baseline

Output (benchmark.in_vivo_dir/huCIRA): huCIRA_BM_Mye_Granulocyte.csv, huCIRA_IL7_{CD4T,CD8T}.csv (sorted by NES)

python 10_benchmark_in_vivo/run_hucira.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hucira as hc
import scanpy as sc

from paper_utils import bench, path

DEFAULT = dict(threshold_lfc=1.0, threshold_pval=0.01, min_size=10, max_size=1000,
               threshold_expression=0.0, permutation_num=1000, threads=6)
RUNS = [  # (dataset, contrast column, (case, control), data cell type, dictionary cell type, output)
    ("gcsf", "condition", ("post", "pre"), "Mye_lineage", "Granulocyte", "huCIRA_BM_Mye_Granulocyte.csv"),
    ("il7", "treatment", ("post_dose1_wk1", "baseline"), "CD4_T", "CD4_T_cell", "huCIRA_IL7_CD4T.csv"),
    ("il7", "treatment", ("post_dose1_wk1", "baseline"), "CD8_T", "CD8_T_cell", "huCIRA_IL7_CD8T.csv"),
]

hcd = hc.load_human_cytokine_dict(save_dir=str(path("reference", "hucira_dir")), force_download=False)
data = {}
for ds, col, contrast, ct, dict_ct, out in RUNS:
    f = bench("in_vivo_dir", "huCIRA", out, new=True)
    if ds not in data:
        data[ds] = sc.read_h5ad(path("inference", ds))  # log-normalized expression in .X
    er = hc.run_one_enrichment_test(adata=data[ds], df=hcd, contrast_combo=contrast, celltype_combo=(ct, dict_ct),
                                    contrast_column=col, celltype_column="celltype", direction="upregulated", **DEFAULT)
    er.sort_values("NES", ascending=False).to_csv(f, index=False)
    print(f"{out}: {len(er)} cytokines, {(er['FDR q-val'] < 0.05).sum()} with FDR < 0.05")
