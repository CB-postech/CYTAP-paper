"""IREA (Immune Dictionary) on the Parse-PBMC benchmark.

Running the IREA web tool for 1,326 conditions is impractical, so its gene-set mode is reproduced
offline: for each condition the up-regulated DEGs (adj. p < 0.05, log FC >= 0.25; RPS/RPL/MT- genes
removed) are tested against every IREA cytokine signature of the matching IREA cell type
(Cytokine_Signatures_Human.xlsx; up-regulated genes) with a hypergeometric test, background = genes
detected in >= 5% of the cells of the Parse cell type. Cytokines are ranked by p (ties ranked
conservatively); for cDC both cDC1 and cDC2 are tried and the better rank kept.
Cytokine names are matched after Greek-letter/punctuation normalisation; Parse cell types without
an IREA counterpart (HSPC, MAIT, NKT, Plasmablast) are 'no_celltype_map'.

Output: IREA_output/irea_top10_results.csv (true_rank, status, ref_size, pool_size)

python 09_benchmark_parse/irea.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from paper_utils import bench, norm_cytokine, path

LFC_CUT, PADJ_CUT, TOP_N = 0.25, 0.05, 10
CT_MAP = {
    "CD4 Memory": "T_cell_CD4", "CD4 Naive": "T_cell_CD4",
    "CD8 Memory": "T_cell_CD8", "CD8 Naive": "T_cell_CD8",
    "Treg": "Treg", "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte",
    "B Naive": "B_cell", "B Intermediate/Memory": "B_cell",
    "NK": "NK_cell", "NK CD56bright": "NK_cell", "ILC": "ILC", "pDC": "pDC",
    "cDC": ["cDC1", "cDC2"],
}
NO_MAP = {"HSPC", "MAIT", "NKT", "Plasmablast"}


def clean_genes(genes):
    return {g for g in genes if g not in ("--", "", None) and not str(g).startswith(("RPS", "RPL", "MT-"))}


pdir = path("benchmark", "parse_dir")
xls = pd.ExcelFile(path("reference", "irea_signatures"))
sig = pd.concat([xls.parse(s) for s in xls.sheet_names], ignore_index=True)
deg = pd.read_csv(pdir / "deg_mapped.csv")
bg = pd.read_csv(pdir / "background_by_celltype.csv")
responsive = pd.read_csv(pdir / "benchmark" / "responsive_conditions.csv")

# IREA gene sets (signature genes are already FDR < 0.05; up-regulated only)
sig_up = sig[(sig["Gene_Human"] != "--") & (sig["Avg_log2FC"] > 0)]
dict_sets = (sig_up.groupby(["Celltype_Str", "Cytokine"])["Gene_Human"]
             .apply(lambda s: clean_genes(set(s.astype(str)))).to_dict())
ct_cyts = sig_up.groupby("Celltype_Str")["Cytokine"].apply(lambda s: sorted(set(s))).to_dict()
RANKABLE = {ict: [ck for ck in cks if dict_sets.get((ict, ck))] for ict, cks in ct_cyts.items()}
irea_celltypes, irea_cytokines = set(sig["Celltype_Str"]), sorted(sig["Cytokine"].unique())

parse_cyts = sorted(responsive["cytokine"].unique())
pn, inx = {norm_cytokine(c): c for c in parse_cyts}, {norm_cytokine(c): c for c in irea_cytokines}
PARSE_TO_IREA = {pn[k]: inx[k] for k in pn.keys() & inx.keys()}

bad = [c for v in CT_MAP.values() for c in (v if isinstance(v, list) else [v]) if c not in irea_celltypes]
if bad:
    sys.exit(f"IREA cell types not found: {bad}")
unknown = set(responsive["celltype"]) - set(CT_MAP) - NO_MAP
if unknown:
    print(f"WARNING: cell types without mapping (scored as no_celltype_map): {sorted(unknown)}")
print(f"cytokines mapped to IREA: {len(PARSE_TO_IREA)}/{len(parse_cyts)}; "
      f"unmapped (not_in_dict): {sorted(set(parse_cyts) - set(PARSE_TO_IREA))}")

deg_f = deg[(deg["cytokine"] != "PBS") & (deg["adj_p_value"] < PADJ_CUT) & (deg["log_fc"] >= LFC_CUT)]
deg_up = {k: clean_genes(set(s["gene"].astype(str))) for k, s in deg_f.groupby(["celltype_adata", "cytokine"])}
N_bg = bg.groupby("celltype")["gene"].apply(lambda s: len(clean_genes(set(s.astype(str))))).to_dict()


def score(parse_ct, cyt):
    base = dict(celltype=parse_ct, cytokine=cyt, true_rank=np.nan, hit=False, ref_size=np.nan, pool_size=np.nan)
    cyt_i = PARSE_TO_IREA.get(cyt)
    if cyt_i is None:
        return {**base, "status": "not_in_dict"}
    ictl = CT_MAP.get(parse_ct)
    if ictl is None:
        return {**base, "status": "no_celltype_map"}
    genes = deg_up.get((parse_ct, cyt))
    if not genes:
        return {**base, "status": "no_geneset"}
    if not N_bg.get(parse_ct):
        return {**base, "status": "no_background"}
    best = None
    for ict in (ictl if isinstance(ictl, list) else [ictl]):
        cands = RANKABLE.get(ict, [])
        if cyt_i not in cands:
            continue
        pv = {ck: hypergeom.sf(len(genes & dict_sets[(ict, ck)]) - 1, N_bg[parse_ct], len(dict_sets[(ict, ck)]),
                               len(genes)) for ck in cands}
        rank = sum(p < pv[cyt_i] for p in pv.values()) + 1
        if best is None or rank < best[0]:
            best = (rank, len(dict_sets[(ict, cyt_i)]), len(cands))
    if best is None:
        return {**base, "status": "not_in_dict"}
    return dict(celltype=parse_ct, cytokine=cyt, true_rank=best[0], hit=bool(best[0] <= TOP_N),
                status="scored", ref_size=best[1], pool_size=best[2])


res = pd.DataFrame([score(r.celltype, r.cytokine) for r in responsive.itertuples()])
res.to_csv(bench("parse_dir", "IREA_output", "irea_top10_results.csv", new=True), index=False)
print(res["status"].value_counts().to_string())
