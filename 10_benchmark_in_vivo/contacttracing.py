"""ContactTracing receptor activity on the in vivo datasets (run inside the ContactTracing docker image).

Pooled analysis with the donor/patient as MAST covariate (with cngeneson): for every receptor of the
ContactTracing interaction list (mouse genes mapped to human with the repository's mapping file; no ligand
filter) expressed in >= 2% of the receiver cells, the MAST interaction test of the ContactTracing repository
(treated vs baseline); receptor activity = number of response genes with interaction FDR < 0.05
(numSigI1_fdr05). Response genes = HVGs (cell_ranger, batch = donor) + all ligands/receptors.
logX = log10(CP10k + 0.1) - log10(0.1) from raw counts.

  gcsf : G-CSF bone marrow, Mye_lineage only (post vs pre), covariate donor
  il7  : IL-7 T cells, CD4_T and CD8_T (post_dose1_wk1 vs baseline), covariate patient

Output (benchmark.in_vivo_dir/ContactTracing_output/{GCSF_BM_mye, IL7_T}/): receptor_activity_ranking.txt, degA.h5ad

  docker run ... -v $PWD:/data <ContactTracing image>      # from the CYTAP-paper root
  cd /data
  python 10_benchmark_in_vivo/contacttracing.py gcsf --njobs 10
  python 10_benchmark_in_vivo/contacttracing.py il7 --njobs 10
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import scale

from paper_utils import bench, path

FRAC_EXP, MIN_CELL = 0.02, 50
RCMD = "MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 R"
DATASETS = {
    "gcsf": dict(file=("data", "gcsf_counts"), counts="raw", cond="condition", case="post", ctrl="pre",
                 sample="donor", covariate="donor", receivers=["Mye_lineage"], subset=True, out="GCSF_BM_mye"),
    "il7": dict(file=("data", "il7_counts"), counts="counts", cond="treatment", case="post_dose1_wk1",
                ctrl="baseline", sample="patient", covariate="patient", receivers=["CD4_T", "CD8_T"], subset=False,
                out="IL7_T"),
}

p = argparse.ArgumentParser()
p.add_argument("dataset", choices=list(DATASETS))
p.add_argument("--njobs", type=int, default=10)
a = p.parse_args()
D = DATASETS[a.dataset]
TUT = path("benchmark", "contacttracing_tutorial")
sys.path.append(str(TUT / "python"))
import contactTracing_library as ct  # noqa: E402

OUT = bench("in_vivo_dir", "ContactTracing_output", D["out"], "x").parent
TMP, INT = OUT / "tmp" / "NOBACKUP", OUT / "interaction_test"
c1, c2 = D["case"], D["ctrl"]

# ---------------------------------------------------------------- data
adata = sc.read(path(*D["file"]))
adata.obs.index.name = adata.var.index.name = None
if D["counts"] == "raw":
    adata.layers["counts"] = adata.raw.X.copy()
adata.obs["cell type"] = adata.obs["celltype"].astype(str)
adata.obs["condition"] = adata.obs[D["cond"]].astype(str)
adata.obs["sample"] = adata.obs[D["sample"]].astype(str)
adata.obs[D["covariate"]] = adata.obs[D["covariate"]].astype(str)
if adata.obs.index.duplicated().any():
    adata.obs_names_make_unique()
clean = lambda s: s.replace(" ", "_").replace("/", "_").replace("#", "")  # noqa: E731
for col in ["condition", "cell type", "sample", D["covariate"]]:
    adata.obs[col] = [clean(x) for x in adata.obs[col]]
keep = adata.obs["condition"].isin([c1, c2])
if D["subset"]:
    keep &= adata.obs["cell type"].isin(D["receivers"])
adata = adata[keep].copy()
print(adata.obs.groupby(["sample", "condition", "cell type"]).size().to_string())

tmp = sc.AnnData(adata.layers["counts"].copy(), obs=adata.obs.copy(), var=adata.var.copy())
sc.pp.normalize_total(tmp, target_sum=1e4)
X = tmp.X.toarray() if issparse(tmp.X) else np.asarray(tmp.X)
adata.layers["logX"] = np.log10(X + 0.1) - np.log10(0.1)
del tmp, X

n = adata.obs[["cell type", "condition"]].value_counts()
small = n[n < MIN_CELL].reset_index()["cell type"].unique()
if len(small):
    print("dropping cell types with <", MIN_CELL, "cells in a condition:", ",".join(small))
    adata = adata[~adata.obs["cell type"].isin(small)].copy()
adata.obs["cngeneson"] = scale(np.asarray((adata.layers["logX"] > 0).sum(axis=1)).flatten())

m2h = pd.read_csv(TUT / "data" / "mouse_to_human_genes.txt", sep="\t")
mapping = dict(zip(m2h["mouseGene"], m2h["humanGene"]))
inter = pd.read_csv(TUT / "data" / "interactions_mm10.txt", sep="\t")
for col in ("ligand", "receptor"):
    inter[col] = inter[col].map(lambda g: mapping.get(g, g.upper()))
inter = inter.drop_duplicates(subset=["ligand", "receptor"])
inter = inter[(inter["ligand"] != inter["receptor"]) & inter["receptor"].isin(set(adata.var.index))]
receptors = list(inter["receptor"].unique())
print(f"{len(receptors)} receptors, {inter['ligand'].nunique()} ligands")

adata.layers["X"] = 10.0 ** (adata.layers["logX"] + np.log10(0.1)) - 0.1
try:
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", batch_key="sample", n_bins=20, max_mean=np.inf, layer="X")
except Exception as e:  # few cells per donor can break the batched HVG
    print("batched HVG failed, retrying without batch_key:", e)
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_bins=20, max_mean=np.inf, layer="X")
del adata.layers["X"]
keep_genes = list(set(adata.var.index[adata.var["highly_variable"]]) | set(inter["receptor"]) | set(inter["ligand"]))
print(f"{len(keep_genes)} genes tested for transcriptional response")

# ---------------------------------------------------------------- interaction tests
os.makedirs(TMP, exist_ok=True)
os.makedirs(INT, exist_ok=True)
genes_present = [g for g in keep_genes if g in adata.var.index]
expected, cmds = set(), []
for rc in D["receivers"]:
    b = adata[adata.obs["cell type"] == rc, genes_present].copy()
    b.var["fracExp"] = np.asarray((b.layers["logX"] > 0).sum(axis=0)).flatten() / b.shape[0]
    b.write(TMP / f"adata_{rc}_inttest.h5ad")
    recs = [r for r in receptors if r in b.var.index and b.var.loc[r, "fracExp"] >= FRAC_EXP]
    f1, f2 = np.asarray(b.obs["condition"] == c1), np.asarray(b.obs["condition"] == c2)
    for g in recs:
        fe = np.asarray(b.layers["logX"][:, np.where(b.var.index == g)[0]] > 0).flatten()
        if fe.sum() < 2 or (~fe).sum() < 2 or not ((fe & f1).any() and (fe & f2).any()
                                                   and (~fe & f1).any() and (~fe & f2).any()):
            continue
        of = INT / rc / g / f"interaction_{c1}_vs_{c2}.txt"
        expected.add(str(of))
        if of.exists():
            continue
        cmds.append(f'mkdir -p {INT}/{rc}/{g} && {RCMD} --vanilla --args --infile "{TMP}/adata_{rc}_inttest.h5ad" '
                    f'--outdir "{INT}/{rc}" --conditions "{c1}" "{c2}" --covariates cngeneson,{D["covariate"]} '
                    f'--numcore 1 --genes "{g}" < {TUT}/R/MAST-interactionTest.R &> {INT}/{rc}/{g}/stdout.txt\n')
(INT / "commands.sh").write_text("".join(cmds))
print(f"{len(cmds)} commands, {len(expected)} expected outputs")
if cmds:
    ct.run_in_background(f"parallel -j {a.njobs} < {INT}/commands.sh", stdoutfile=f"{INT}/commands_stdout.txt", wait=True)
print(f"done {sum(os.path.exists(f) for f in expected)}/{len(expected)}")

deg = ct.read_all_contactTracing_results(D["receivers"], receptors, str(INT), cond1=c1, cond2=c2,
                                         inttest=f"interaction_{c1}_vs_{c2}", ncore=10, genes=keep_genes)
deg.obs["numSigI1_fdr05"] = np.sum(deg.layers["fdr.i1"] < 0.05, axis=1)
ranking = deg.obs.rename(columns={"cell type": "cell_type_receptor"}).sort_values("numSigI1_fdr05", ascending=False)
ranking.to_csv(OUT / "receptor_activity_ranking.txt", sep="\t")
deg.write(OUT / "degA.h5ad")
print(ranking[["cell_type_receptor", "receptor", "numSigI1_fdr05"]].head(20).to_string())
