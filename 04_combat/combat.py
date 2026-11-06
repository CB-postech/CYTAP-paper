"""COMBAT (unpaired): CYTAP logits vs COVID-19 severity, and plasma IL-6 (Luminex).

Donors of HV, COVID_MILD, COVID_SEV and COVID_CRIT (one sample per donor). Severity trend: donor median logit
vs severity (0-3), Spearman rho with BH-FDR over the 90 cytokines (cytap.stats.severity_trend), for both
control references (pooled healthy-donor controls; CellOT counterfactual controls). Cell-type level: donor x
cell type groups with > 100 cells.

Outputs (results/combat/), {ref} = pooled | cellot:
  combat_{ref}_severity_cytokine_allcell.csv        increasing cytokines (slope > 0, FDR < 0.05), top 20 by rho
  combat_{ref}_severity_cytokine_allcell_full.csv   all 90 cytokines
  combat_{ref}_severity_cytokine_by_celltype.csv    same test per cell type
  combat_{ref}_{IL_32_beta,IL_6}.csv                donor x cell type median logit
  il6_plasma_by_severity.csv     plasma IL-6 per donor (one sample per donor; below detection removed)
  il6_plasma_vs_logit.csv        plasma IL-6 vs sample median CYTAP IL-6 logit (CellOT reference, all cells)
  il6_receptor_by_celltype.csv   donor x cell type median IL-6 logit and mean IL6R / IL6ST / IL6 expression
  il6_signature_by_celltype.csv  donor x cell type median IL-6/JAK/STAT3 signature score

python 04_combat/combat.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

import cytap
from paper_utils import filter_groups, out, path, response_table, trend_table

SRC, DONOR, CT, SAMPLE = "Source", "donor_id", "major_subset", "scRNASeq_sample_ID"
ORDER = ["HV", "COVID_MILD", "COVID_SEV", "COVID_CRIT"]
CT_ORDER = ["B", "CD4", "CD8", "cMono", "DP", "ncMono", "NK"]
MIN_CELLS = 100


def load(ref):
    adata = sc.read_h5ad(path("inference", f"combat_{ref}"))
    adata = adata[adata.obs[SRC].isin(ORDER)].copy()
    adata.obs[SRC] = adata.obs[SRC].cat.remove_unused_categories()
    return adata


# ---------------------------------------------------------------- severity trend
for ref in ["pooled", "cellot"]:
    adata = load(ref)
    print(ref, adata.n_obs, "cells", adata.obs.groupby(SRC, observed=True)[DONOR].nunique().to_dict())
    res = cytap.stats.severity_trend(adata, SRC, ORDER, DONOR)
    res.to_csv(out("combat", f"combat_{ref}_severity_cytokine_allcell_full.csv"), index=False)
    trend_table(res).to_csv(out("combat", f"combat_{ref}_severity_cytokine_allcell.csv"), index=False)

    ad_ct = filter_groups(adata, [DONOR, CT], MIN_CELLS)
    per_ct = []
    for ct in ad_ct.obs[CT].cat.categories:
        try:
            r = cytap.stats.severity_trend(ad_ct, SRC, ORDER, DONOR, celltype_key=CT, celltype=ct)
        except ValueError as e:  # a severity group without donors
            print(f"  skip {ct}: {e}")
            continue
        per_ct.append(r.assign(cell_type=ct))
    pd.concat(per_ct).to_csv(out("combat", f"combat_{ref}_severity_cytokine_by_celltype.csv"), index=False)

    for name, cyto in {"IL_32_beta": "IL-32-beta", "IL_6": "IL-6"}.items():
        pb = cytap.stats.pseudobulk(ad_ct, DONOR, SRC, CT, cytokines=[cyto], conditions=ORDER,
                                    require_all_conditions=True)
        response_table(pb, cyto, DONOR, SRC, CT).to_csv(out("combat", f"combat_{ref}_{name}.csv"), index=False)

# ---------------------------------------------------------------- plasma IL-6 (Luminex)
LX_GROUPS = {"HV": 0, "COVID-mild": 1, "COVID-severe": 2, "COVID-critical": 3}
LX_META = ["Row", "severity", "sex", "age", "BMI", "dexamethasone"]


def sample_key(s):  # "S00029-Ja005E-PBCa" -> "S00029-Ja005E"
    return s.astype(str).str.split("-").str[:2].str.join("-")


lx = pd.read_excel(path("data", "combat_luminex"), sheet_name="All data")
lx = lx[lx["severity"].isin(LX_GROUPS)].copy()
names = {c: re.sub(r"\s*\(BR.*$", "", c).strip() for c in lx.columns if c not in LX_META}
lx = lx.rename(columns=names)
prots = list(names.values())
lx[prots] = lx[prots].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)  # 0 = below detection
lx["key"] = sample_key(lx["Row"])
lx = lx[lx["key"].str.count("-") == 1].set_index("key")  # drop HV000x / control plate rows
lx["sev"] = lx["severity"].map(LX_GROUPS)

# one sample per donor (most severe, then latest visit), as for scRNA-seq
lxd = lx.assign(donor=lx.index.str.split("-").str[0], visit=lx.index.str.split("-").str[1])
e = lxd.sort_values(["donor", "sev", "visit"], ascending=[True, False, False]).drop_duplicates("donor", keep="first")
e = e[["severity", "sev", "IL-6"]].rename(columns={"IL-6": "plasma"}).dropna(subset=["plasma"])
e.to_csv(out("combat", "il6_plasma_by_severity.csv"))
rho, p = spearmanr(e["plasma"], e["sev"])
print(f"plasma IL-6 vs severity: n={len(e)}, rho={rho:.3f}, P={p:.2e}")

# matched samples: median IL-6 logit over all cells of the sample
cel = load("cellot")
lg = cytap.stats.logits(cel)
lg["key"] = sample_key(cel.obs[SAMPLE]).values
per_sample = lg.groupby("key").median()
shared = per_sample.index.intersection(lx.index)
f = pd.DataFrame({"sample": shared, "severity": lx.loc[shared, "severity"].values,
                  "plasma": lx.loc[shared, "IL-6"].astype(float).values,
                  "logit": per_sample.loc[shared, "IL-6"].values}).dropna(subset=["plasma"])
f.to_csv(out("combat", "il6_plasma_vs_logit.csv"), index=False)
rho, p = spearmanr(f["plasma"], f["logit"])
print(f"plasma IL-6 vs IL-6 logit: n={len(f)} of {len(shared)} matched, rho={rho:.3f}, P={p:.2e}")

# ---------------------------------------------------------------- IL-6 receptors and signature (expression)
ex = sc.read_h5ad(path("data", "combat_expression"))
ex = ex[ex.obs[SRC].isin(ORDER)].copy()
assert set(ex.obs_names) == set(cel.obs_names), "expression and CYTAP objects contain different cells"
ex = ex[cel.obs_names].copy()
ex.obsm["pred_probs"], ex.obsm["pred_logits"] = cel.obsm["pred_probs"], cel.obsm["pred_logits"]
ex.uns["pred_classes"] = cel.uns["pred_classes"]

# donor x cell type groups > MIN_CELLS, cell types present in all four groups
ad_ct = filter_groups(ex, [DONOR, CT], MIN_CELLS)
present = ad_ct.obs.groupby([CT, SRC], observed=True).size().unstack(fill_value=0) > 0
ad_ct = ad_ct[ad_ct.obs[CT].isin(present.index[present[ORDER].all(axis=1)])].copy()
keys = [CT, DONOR, SRC]
pb = cytap.stats.pseudobulk(ad_ct, DONOR, SRC, CT, cytokines=["IL-6"])
genes = [g for g in ["IL6R", "IL6ST", "IL6"] if g in ad_ct.var_names]
X = ad_ct[:, genes].X
gx = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else np.asarray(X), columns=genes)
gx[keys] = ad_ct.obs[keys].astype(str).values
gx = gx.groupby(keys, observed=True)[genes].mean().reset_index()  # mean: receptor expression is sparse
for k in keys:
    pb[k] = pb[k].astype(str)
long = pd.concat([pb.melt(id_vars=keys, value_vars=["IL-6"], var_name="panel", value_name="value"),
                  gx.melt(id_vars=keys, value_vars=genes, var_name="panel", value_name="value")])
long = long[long[CT].isin(CT_ORDER)].rename(columns={CT: "cell_type", DONOR: "donor", SRC: "condition"})
long.to_csv(out("combat", "il6_receptor_by_celltype.csv"), index=False)

# IL-6/JAK/STAT3 signature (scored on all cells of the four groups, then filtered as above)
SCORE = "H_IL6_JAK_STAT3"
sig = [g for g in gp.get_library("MSigDB_Hallmark_2020")["IL-6/JAK/STAT3 Signaling"] if g in ex.var_names]
print(f"{SCORE}: {len(sig)} genes")
sc.tl.score_genes(ex, gene_list=sig, score_name=SCORE, use_raw=False)
n = ex.obs.groupby([DONOR, CT], observed=True)[DONOR].transform("size")
d = ex.obs.loc[(n > MIN_CELLS) & ex.obs[CT].isin(CT_ORDER), keys + [SCORE]].copy()
for k in keys:
    d[k] = d[k].astype(str)
(d.groupby(keys, observed=True)[SCORE].median().reset_index()
   .rename(columns={CT: "cell_type", DONOR: "donor", SRC: "condition", SCORE: "score"})
   .to_csv(out("combat", "il6_signature_by_celltype.csv"), index=False))
