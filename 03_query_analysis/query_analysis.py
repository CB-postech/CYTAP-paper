"""CYTAP statistics, signature scores and UMAPs of the paired query datasets (input: 02_inference outputs).

Condition comparisons use cytap.stats.compare_conditions (donor-level median logits, paired one-sided t-test,
BH-FDR over the 90 cytokines); within-donor changes use cytap.stats.paired_delta. Signature scores: scanpy
score_genes on the log-normalized expression.

  ifnb        IFN-beta PBMC: stim vs ctrl (all cells, megakaryocytes), IFN-beta logit per cell type, GO
              IFN-beta score vs IFN-beta logit
  macrophage  IFNg+/IFNb+/IL4+/TNFa+ vs Control; IFN-beta/IFN-gamma/IL-4/TNF-alpha logit change per condition;
              signature scores (fibroblast co-culture excluded)
  il7         week 1 vs baseline in CD4 and CD8 T cells
  gcsf        post vs pre per lineage (2 donors: ranked by effect size)

Outputs: results/{ifnb,macrophage,il7,gcsf}/*.csv, *.png (State-embedding UMAPs)

python 03_query_analysis/query_analysis.py [ifnb macrophage il7 gcsf]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

import cytap
from paper_utils import compare_table, delta_table, out, path, response_table

GO_BP = "GO_Biological_Process_2026"
HALLMARK = "MSigDB_Hallmark_2020"


def state_umap(adata):
    """UMAP of the State embedding; returns its obsm key (scanpy versions store it as state_umap or X_state_umap)."""
    sc.pp.neighbors(adata, use_rep="X_state")
    sc.tl.umap(adata, key_added="state_umap")
    return "X_state_umap" if "X_state_umap" in adata.obsm else "state_umap"


def save_umap(adata, folder, color, fname, **kw):
    sc.pl.embedding(adata, basis="state_umap", color=color, frameon=False, use_raw=False, show=False, **kw)
    plt.savefig(out(folder, fname), dpi=300, bbox_inches="tight")
    plt.close()


def score(adata, genes, name):
    genes = [g for g in genes if g in adata.var_names]
    print(f"{name}: {len(genes)} genes")
    sc.tl.score_genes(adata, gene_list=genes, score_name=name, use_raw=False)


def ifnb():
    COND, CTRL, STIM, DONOR, CT, CYTO = "stim", "ctrl", "stim", "ind", "cell", "IFN-beta"
    adata = sc.read_h5ad(path("inference", "ifnb"))
    kw = dict(donor_key=DONOR, celltype_key=CT, paired=True)
    compare_table(cytap.stats.compare_conditions(adata, COND, CTRL, STIM, **kw)).to_csv(
        out("ifnb", "IFNb_top_cytokine.csv"), index=False)
    compare_table(cytap.stats.compare_conditions(adata, COND, CTRL, STIM, celltype="Megakaryocytes", **kw)).to_csv(
        out("ifnb", "IFNb_Megakaryocytes_top_cytokine.csv"), index=False)
    pb = cytap.stats.pseudobulk(adata, DONOR, COND, CT, cytokines=[CYTO])
    response_table(pb, CYTO, DONOR, COND, CT).to_csv(out("ifnb", "IFNb_IFN_beta.csv"), index=False)
    dl = cytap.stats.paired_delta(adata, CYTO, COND, CTRL, STIM, DONOR, CT)
    delta_table(dl).to_csv(out("ifnb", "IFNb_IFN_beta_delta.csv"), index=False)

    score(adata, gp.get_library(GO_BP)["Response to Interferon-Beta (GO:0035456)"], "GOBP_IFN_beta_score")
    cells = pd.DataFrame({
        "Signature_Score": adata.obs["GOBP_IFN_beta_score"].values,
        "Predicted_Logit": cytap.stats.logits(adata)[CYTO].to_numpy(),
        "Cell_Type": adata.obs[CT].values,
        "Condition": adata.obs[COND].values,
        "Donor": adata.obs[DONOR].values,
    })
    cells.to_csv(out("ifnb", "IFNb_pseudobulk_correlation_data.csv"), index=False)
    med = cells.groupby(["Donor", "Cell_Type", "Condition"], observed=True)[["Signature_Score", "Predicted_Logit"]].median()
    rho, p = spearmanr(med["Signature_Score"], med["Predicted_Logit"], nan_policy="omit")
    print(f"GO IFN-beta score vs IFN-beta logit (donor x cell type x condition medians): n={len(med)}, "
          f"Spearman rho={rho:.3f}, P={p:.2e}")

    umap_key = state_umap(adata)
    adata.uns[f"{COND}_colors"] = ["#219ebc", "#fb8500"]
    adata.uns[f"{CT}_colors"] = ["#0077b6", "#ff9500", "#4f772d", "#9e2a2b", "#7b2cbf", "#c18c5d", "#fb6f92", "#6c757d"]
    for g in ["IFNAR1", "IFNAR2", "ISG15", "IFIT1"]:
        save_umap(adata, "ifnb", g, f"{g}_expression.png", cmap="Reds", size=20)
    save_umap(adata, "ifnb", COND, "IFNb_stim_umap.png", size=10)
    save_umap(adata, "ifnb", CT, "IFNb_celltype_umap.png", size=10)
    save_umap(adata, "ifnb", "GOBP_IFN_beta_score", "GOBP_IFN_beta_score_umap.png", cmap="Reds", size=15)
    ax = cytap.plotting.embedding(adata, CYTO, basis=umap_key)
    ax.figure.savefig(out("ifnb", "IFNb_IFN_beta_logit_umap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def macrophage():
    COND, CTRL, DONOR = "condition", "Control", "sample"
    STIMS = {"IFNg": "IFNg+", "IFNb": "IFNb+", "IL4": "IL4+", "TNFa": "TNFa+"}
    TARGETS = ["IFNb+", "IFNg+", "IFNg+ fibro+", "IL4+", "TNFa+", "TNFa+ IFNg+", "TNFa+ IFNg+ fibro+", "TNFa+ fibro+"]
    DELTA = {"IFN_gamma": "IFN-gamma", "IFN_beta": "IFN-beta", "IL_4": "IL-4", "TNF_alpha": "TNF-alpha"}
    adata = sc.read_h5ad(path("inference", "macrophage"))
    for name, stim in STIMS.items():   # one cell type: donor-level paired test on all cells
        res = cytap.stats.compare_conditions(adata, COND, CTRL, stim, donor_key=DONOR, paired=True)
        compare_table(res).to_csv(out("macrophage", f"Mac_{name}_top_cytokine.csv"), index=False)
    for name, cyto in DELTA.items():
        dl = cytap.stats.paired_delta(adata, cyto, COND, CTRL, TARGETS, DONOR)
        delta_table(dl).to_csv(out("macrophage", f"Mac_{name}_delta.csv"), index=False)

    sub = adata[~adata.obs["stim"].str.contains("fibro", na=False, case=False)].copy()   # no fibroblast co-culture
    umap_key = state_umap(sub)
    sub.uns[f"{COND}_colors"] = ["#0286df", "#a5be00", "#ffba08", "#09989b", "#735639", "#fa80d1"]
    save_umap(sub, "macrophage", COND, "Mac_condition_umap.png", size=5)
    ax = cytap.plotting.embedding(sub, "IL-4", basis=umap_key, s=30)
    ax.figure.savefig(out("macrophage", "state_umap_logit_IL-4.png"), dpi=300, bbox_inches="tight")
    plt.close()

    go_bp, hallmark = gp.get_library(GO_BP), gp.get_library(HALLMARK)
    signatures = {
        "IFN_beta": ("GOBP_IFN_beta_score", go_bp["Response to Interferon-Beta (GO:0035456)"]),
        "IFN_gamma": ("H_IFN_gamma_score", hallmark["Interferon Gamma Response"]),
        "IL4": ("GOBP_IL4_score", go_bp["Response to Interleukin-4 (GO:0070670)"]),
        "TNFa": ("H_TNFa_score", hallmark["TNF-alpha Signaling via NF-kB"]),
    }
    for name, (col, genes) in signatures.items():
        score(sub, genes, col)
        d = sub.obs[[DONOR, "stim", col]].copy()
        d.insert(0, "Group", "All Cells")
        (d.groupby(["Group", DONOR, "stim"], observed=True)[col].median().reset_index().dropna()
           .rename(columns={DONOR: "Donor", "stim": "Condition", col: "Value"})
           .to_csv(out("macrophage", f"Mac_{name}_signature_score.csv"), index=False))


def il7():
    COND, CTRL, STIM, DONOR, CT = "treatment", "baseline", "post_dose1_wk1", "patient", "celltype"
    adata = sc.read_h5ad(path("inference", "il7"))
    for name, ct in {"CD4T": "CD4_T", "CD8T": "CD8_T"}.items():
        res = cytap.stats.compare_conditions(adata, COND, CTRL, STIM, donor_key=DONOR, celltype_key=CT,
                                             celltype=ct, paired=True)
        compare_table(res).to_csv(out("il7", f"IL7_tcell_{name}_top_cytokine.csv"), index=False)
    pb = cytap.stats.pseudobulk(adata, DONOR, COND, cytokines=["IL-7"])
    response_table(pb, "IL-7", DONOR, COND).to_csv(out("il7", "IL7_tcell_IL_7.csv"), index=False)


def gcsf():
    COND, CTRL, STIM, DONOR, CT = "condition", "pre", "post", "donor", "celltype"
    adata = sc.read_h5ad(path("inference", "gcsf"))
    for ct in ["Mye_lineage", "T_NK", "HSPC", "B_lineage", "EM_lineage"]:
        res = cytap.stats.compare_conditions(adata, COND, CTRL, STIM, donor_key=DONOR, celltype_key=CT,
                                             celltype=ct, paired=True)
        compare_table(res).to_csv(out("gcsf", f"GCSF_BM_{ct}_top_cytokine.csv"), index=False)
    dl = cytap.stats.paired_delta(adata, "G-CSF", COND, CTRL, STIM, DONOR, CT)
    delta_table(dl).to_csv(out("gcsf", "GCSF_BM_G_CSF_delta.csv"), index=False)
    state_umap(adata)
    save_umap(adata, "gcsf", "CSF3R", "CSF3R_expression.png", cmap="Reds", size=20)


DATASETS = {"ifnb": ifnb, "macrophage": macrophage, "il7": il7, "gcsf": gcsf}
for name in sys.argv[1:] or DATASETS:
    print(f"== {name}")
    DATASETS[name]()
