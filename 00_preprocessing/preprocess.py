"""Query and training data preparation (Python steps). Run from the repository root.

python 00_preprocessing/preprocess.py normalize IN.h5ad OUT.h5ad [--counts X|raw|LAYER]
    normalize_total(1e4) + log1p, input of the State embedding:
      Parse-PBMC  parse.counts -> parse.normalized
      G-CSF       data.gcsf_counts -> data/normalized/bm_normalized.h5ad  --counts raw
      macrophage  data/normalized/macrophage_stim_counts.h5ad -> data/normalized/macrophage_stim_normalized.h5ad
      IL-7        counts of each cohort -> data/normalized/IL7_tcell_{CRC,GBM}.h5ad
python 00_preprocessing/preprocess.py macrophage DIR
    counts / metadata / genes written by preprocess.R macrophage -> data/normalized/macrophage_stim_counts.h5ad
python 00_preprocessing/preprocess.py combat DOWNLOADED.h5ad
    COMBAT (CZ CELLxGENE): gene symbols as var_names (Ensembl IDs in var['ensembl_id']); cells without a
    major_subset annotation (doublets, red blood cells) removed -> data.combat_expression_unfiltered
python 00_preprocessing/preprocess.py combat-select {state_emb,expression}
    COMBAT cell and sample selection, identical for the State-embedding object (CYTAP input) and the expression
    object: cell types with < 500 cells (iNKT, RET, Mast) and samples with <= 500 cells removed; one sample per
    donor (most severe, then latest time point) -> data.combat / data.combat_expression
python 00_preprocessing/preprocess.py parse-hvg
    HVG input of the input ablation: 2,058 HVGs (= State embedding dimension) of the log-normalized Parse-PBMC data
    (batch_key = donor, all donors); cytokine-treated cells -> parse.hvg_npz, parse.hvg_genes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from paper_utils import WORK, path


def new(p: Path) -> Path:
    if p.exists():
        sys.exit(f"{p} exists")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def normalize(a):
    dst = new(Path(a.output))
    adata = sc.read_h5ad(a.input)
    if a.counts == "raw":
        adata.X = adata.raw.X.copy()
    elif a.counts != "X":
        adata.X = adata.layers[a.counts].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.write_h5ad(dst)


def macrophage(a):
    import anndata as ad
    import scipy.io
    d = Path(a.dir)
    dst = new(WORK / "data/normalized/macrophage_stim_counts.h5ad")
    X = sp.csr_matrix(scipy.io.mmread(d / "macrophage_stim_counts.mtx").T)        # genes x cells -> cells x genes
    obs = pd.read_csv(d / "macrophage_stim_metadata.csv")
    genes = pd.read_csv(d / "macrophage_stim_gene_names.csv", header=None)[0].astype(str)
    adata = ad.AnnData(X=X, obs=obs.set_index(obs["barcode"].astype(str).values), var=pd.DataFrame(index=genes.values))
    adata.var_names_make_unique()
    adata.obs["condition"] = adata.obs["stim"]   # control label "Control"
    adata.write_h5ad(dst)


def combat(a):
    dst = new(path("data", "combat_expression_unfiltered"))
    adata = sc.read_h5ad(a.input)
    adata.var = adata.var.reset_index().rename(columns={"index": "ensembl_id"}).set_index("feature_name")
    adata.var.index = adata.var.index.astype(str)
    adata.var.index.name = None
    adata.var_names_make_unique()
    adata = adata[adata.obs["major_subset"].astype(str) != "nan"].copy()
    adata.write_h5ad(dst)


COMBAT_EXCLUDE_CELLTYPES = ["iNKT", "RET", "Mast"]
COMBAT_EXCLUDE_SAMPLES = ["U00504-Ua005E-PBUa", "G05092-Ja005E-PBCa", "S00030-Ja003E-PBCa"]   # <= 500 cells
COMBAT_PRIORITY = {"COVID_CRIT": 1, "COVID_SEV": 2, "COVID_MILD": 3}  # others: 3


def combat_select(a):
    key = {"state_emb": "combat", "expression": "combat_expression"}[a.which]
    dst = new(path("data", key))
    adata = sc.read_h5ad(path("data", f"{key}_unfiltered"))
    n0 = adata.n_obs
    adata = adata[~adata.obs["major_subset"].isin(COMBAT_EXCLUDE_CELLTYPES)
                  & ~adata.obs["scRNASeq_sample_ID"].isin(COMBAT_EXCLUDE_SAMPLES)].copy()
    meta = adata.obs[["donor_id", "scRNASeq_sample_ID", "Source"]].drop_duplicates().copy()
    meta["priority"] = meta["Source"].map(COMBAT_PRIORITY).astype(float).fillna(3)
    meta["time_part"] = meta["scRNASeq_sample_ID"].astype(str).str.split("-").str[1]
    keep = (meta.sort_values(["donor_id", "priority", "time_part"], ascending=[True, True, False])
                .drop_duplicates("donor_id", keep="first")["scRNASeq_sample_ID"])
    adata = adata[adata.obs["scRNASeq_sample_ID"].isin(keep)].copy()
    for c in ["major_subset", "scRNASeq_sample_ID"]:
        if isinstance(adata.obs[c].dtype, pd.CategoricalDtype):
            adata.obs[c] = adata.obs[c].cat.remove_unused_categories()
    assert adata.obs["scRNASeq_sample_ID"].nunique() == adata.obs["donor_id"].nunique()
    print(f"{n0} -> {adata.n_obs} cells, {adata.obs['donor_id'].nunique()} donors")
    adata.write_h5ad(dst)


def parse_hvg(a):
    dst, genes_out = new(path("parse", "hvg_npz")), new(path("parse", "hvg_genes"))
    adata = sc.read_h5ad(path("parse", "emb"))   # log-normalized expression in .X
    sc.pp.highly_variable_genes(adata, n_top_genes=2058, subset=True, batch_key="donor")
    genes_out.write_text("\n".join(adata.var_names) + "\n")
    treated = (adata.obs["cytokine"].astype(str) != "PBS").to_numpy()
    X = adata.X[treated]
    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
    classes, y = np.unique(adata.obs["cytokine"].astype(str).to_numpy()[treated], return_inverse=True)
    np.savez_compressed(dst, X=X, y=y.astype(np.int64), donor=adata.obs["donor"].astype(str).to_numpy()[treated],
                        cell_type=adata.obs["cell_type"].astype(str).to_numpy()[treated],
                        gene_names=adata.var_names.values, classes=classes)
    print(f"saved {X.shape} to {dst}")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
q = sub.add_parser("normalize")
q.add_argument("input")
q.add_argument("output")
q.add_argument("--counts", default="X", help="X, raw (adata.raw.X) or a layer name")
q.set_defaults(func=normalize)
q = sub.add_parser("macrophage")
q.add_argument("dir")
q.set_defaults(func=macrophage)
q = sub.add_parser("combat")
q.add_argument("input")
q.set_defaults(func=combat)
q = sub.add_parser("combat-select")
q.add_argument("which", choices=["state_emb", "expression"])
q.set_defaults(func=combat_select)
sub.add_parser("parse-hvg").set_defaults(func=parse_hvg)
a = p.parse_args()
a.func(a)
