"""Paths, input representations and helpers shared by the analysis scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


CFG = yaml.safe_load(open(ROOT / "config" / "paths.yaml"))
if (ROOT / "config" / "paths.local.yaml").exists():  # machine-specific entries (not tracked)
    CFG = _merge(CFG, yaml.safe_load(open(ROOT / "config" / "paths.local.yaml")) or {})
WORK = Path(os.environ.get("CYTAP_WORK_DIR", CFG["work_dir"]))  # override, e.g. inside docker
if not WORK.is_absolute():
    WORK = (ROOT / WORK).resolve()


def path(*keys) -> Path:
    """path('inference', 'ifnb') -> absolute path from config/paths.yaml."""
    v = CFG
    for k in keys:
        v = v[k]
    p = Path(v)
    return p if p.is_absolute() else WORK / p


def out(*parts) -> Path:
    """File under the results folder (parent folders are created)."""
    p = path("results").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def bench(kind: str, *parts, new: bool = False) -> Path:
    """File under benchmark.parse_dir / benchmark.in_vivo_dir (parent folders created).
    new=True: stop if the file exists (FORCE=1 to replace)."""
    p = path("benchmark", kind).joinpath(*parts)
    if new and p.exists() and os.environ.get("FORCE") != "1":
        sys.exit(f"{p} exists (set FORCE=1 to replace)")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- result tables (column names used in the paper tables)
def compare_table(res: pd.DataFrame) -> pd.DataFrame:
    s, c = res.attrs["stim"], res.attrs["ctrl"]
    return res.rename(columns={
        "cytokine": "Cytokine",
        "mean_logit_stim": f"Mean_Logit ({s})",
        "mean_logit_ctrl": f"Mean_Logit ({c})",
        "delta_logit": "Effect_Size (Delta Logit)",
        "log2fc_prob": "Log2FC (Probability)",
        "pval": "T-test_P",
        "fdr": "T-test_FDR",
    })


def trend_table(res: pd.DataFrame, fdr: float = 0.05, top_n: int | None = 20) -> pd.DataFrame:
    """Increasing cytokines (slope > 0, FDR < fdr) ranked by Spearman rho."""
    d = res[(res["slope"] > 0) & (res["fdr"] < fdr)].sort_values("rho", ascending=False)
    if top_n:
        d = d.head(top_n)
    return d.rename(columns={"cytokine": "Cytokine", "rho": "Correlation_Rho", "slope": "Regression_Slope",
                             "pval": "P_value", "fdr": "FDR"}).drop(columns="n_donors")


def response_table(pb: pd.DataFrame, cytokine: str, donor_key, condition_key, celltype_key=None) -> pd.DataFrame:
    d = pb.rename(columns={donor_key: "Donor", condition_key: "Condition", cytokine: "Value"})
    if celltype_key:
        d = d.rename(columns={celltype_key: "Group"})
    else:
        d.insert(0, "Group", "All Cells")
    return d[["Group", "Donor", "Condition", "Value"]]


def delta_table(dl: pd.DataFrame) -> pd.DataFrame:
    return dl.rename(columns={"group": "Group", "donor": "Donor", "condition": "Target_Condition", "delta": "Delta"})


def filter_groups(adata, keys, min_cells: int):
    """Keep cells in groups (e.g. donor x cell type) with more than ``min_cells`` cells."""
    n = adata.obs.groupby(keys, observed=True)[keys[0]].transform("size")
    a = adata[(n > min_cells).to_numpy()].copy()
    for k in keys:
        if isinstance(a.obs[k].dtype, pd.CategoricalDtype):
            a.obs[k] = a.obs[k].cat.remove_unused_categories()
    return a


# ---------------------------------------------------------------- input representations (input ablation)
# cytap uses the Comp input only; the alternatives are defined here.
#   Raw = z,  Adj = z - (c_dk - c_k),  Dev = z - c_dk,  Comp = [Adj ; Dev]  (= cytap.features.compose)
# Names in the result tables and CV model files: Raw = Baseline, Adj = Donor_Only, Dev = Sub, Comp = Donor_Sub,
# HVG = HVG_Baseline.
REPRESENTATIONS = ("raw", "adj", "dev", "comp")
TABLE_NAMES = {"raw": "Baseline", "adj": "Donor_Only", "dev": "Sub", "comp": "Donor_Sub", "hvg": "HVG_Baseline"}


def compose_input(z, ck, cdk, representation: str = "comp"):
    """Input representation from row-matched z, c_k, c_{d,k}."""
    if representation == "raw":
        return z
    if representation == "adj":
        return z - (cdk - ck)
    if representation == "dev":
        return z - cdk
    if representation == "comp":
        from cytap.features import compose
        return compose(z, ck, cdk)
    raise ValueError(f"representation must be one of {REPRESENTATIONS}")


def reference_inputs(X, condition, donor, celltype, control_label="PBS", representation="comp"):
    """cytap.train.build_reference_inputs for any input representation (cytokine-treated Parse cells)."""
    from cytap.features import compute_centroids
    condition = np.asarray(condition).astype(str)
    is_ctrl = condition == control_label
    cents = compute_centroids(X, is_ctrl, donor=donor, celltype=celltype, fallback=True)
    treated = ~is_ctrl[cents.keep]
    rows = np.flatnonzero(cents.keep)[treated]
    ck, cdk = cents.rows(treated)
    X_in = compose_input(np.asarray(X)[rows], ck, cdk, representation)
    classes, y = np.unique(condition[rows], return_inverse=True)
    return X_in, y.astype(np.int64), classes, np.asarray(donor)[rows], np.asarray(celltype)[rows]


def predict_input(adata, model, control_key, control_label, donor_key, celltype_key=None, batch_size=4096):
    """cytap.predict(design="paired") for a model trained on another input representation
    (model.config['representation']): same cell filtering, centroids and outputs."""
    from cytap.features import compute_centroids
    from cytap.predict import predict_proba
    from cytap.stats import _logit
    rep, embed_key = model.config["representation"], model.config["embedding_key"]
    X = np.asarray(adata.obsm[embed_key])
    is_ctrl = (adata.obs[control_key] == control_label).to_numpy()
    cents = compute_centroids(X, is_ctrl, donor=adata.obs[donor_key].to_numpy(),
                              celltype=adata.obs[celltype_key].to_numpy() if celltype_key else None)
    Xk = X[cents.keep]
    probs = []
    for i in range(0, len(Xk), batch_size):
        sl = slice(i, i + batch_size)
        ck, cdk = cents.rows(sl)
        probs.append(predict_proba(model, compose_input(Xk[sl], ck, cdk, rep), batch_size))
    probs = np.vstack(probs)
    res = adata[cents.keep].copy()
    res.obsm["pred_probs"], res.obsm["pred_logits"] = probs, _logit(probs)
    res.uns["pred_classes"] = np.asarray(model.classes)
    return res


# ---------------------------------------------------------------- Parse-PBMC cross-validation
FOLD_DONORS = {1: ["Donor10", "Donor11", "Donor12"], 2: ["Donor7", "Donor8", "Donor9"],
               3: ["Donor4", "Donor5", "Donor6"], 4: ["Donor1", "Donor2", "Donor3"]}


def load_parse(embed_key: str = "X_state"):
    """State embeddings and (donor, cell_type, cytokine) labels of all Parse-PBMC cells."""
    import anndata as ad
    a = ad.read_h5ad(path("parse", "emb"), backed="r")
    Z = np.array(a.obsm[embed_key][:], dtype=np.float32)
    obs = a.obs[["donor", "cell_type", "cytokine"]].astype(str).reset_index(drop=True)
    return Z, obs


def load_hvg():
    """HVG input of the ablation (00_preprocessing/preprocess.py parse-hvg): X, y, donor, cell type, classes."""
    with np.load(path("parse", "hvg_npz"), allow_pickle=True) as d:
        X, y, donor, ct = d["X"], d["y"], d["donor"].astype(str), d["cell_type"].astype(str)
        if "classes" in d.files:
            classes = d["classes"]
        else:  # file without class names: sorted cytokines, as in the CYTAP model
            import torch
            classes = np.array(torch.load(path("model"), map_location="cpu", weights_only=True)["classes"])
    return X, y, donor, ct, classes


def cellot_map(run, target="heldout"):
    """cytap CellOT map of a Parse-PBMC run (1-4, 'nofold') or 'combat' (05_cellot/cellot_maps.py export).

    target (folds 1-4): PBS cells the map was trained onto, of the held-out donors ('heldout') or of the training
    donors ('train')."""
    from cytap.counterfactual import CellOT
    runs = CFG["cellot"]["parse_runs_trainctrl" if target == "train" else "parse_runs"]
    name = "combat" if run == "combat" else runs[run if run == "nofold" else int(run)]
    return CellOT.load(path("cellot", "maps") / f"{name}_cellot.pt")


def counterfactual_centroids(cot, Z, rows, keys: pd.MultiIndex, source="cytokine", chunk=200_000,
                             dtype="float32"):
    """{key: mean of the counterfactual controls of cells Z[rows]} for one CellOT map.

    Cells are encoded with the map's PCA, transported by the map of ``source``, decoded to the embedding and
    averaged per key (one key per row, e.g. (donor, cell type))."""
    import scipy.sparse as sp
    codes, uniq = pd.factorize(keys)
    S = np.zeros((len(uniq), Z.shape[1]))
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        cf = cot.decode(cot.transport_pcs(cot.encode(Z[r]), source))
        M = sp.csr_matrix((np.ones(len(r)), (codes[s:s + chunk], np.arange(len(r)))), shape=(len(uniq), len(r)))
        S += M @ cf.astype(np.float64)
    N = np.bincount(codes, minlength=len(uniq))
    return {tuple(u): (S[i] / N[i]).astype(dtype) for i, u in enumerate(uniq)}


def fold_references(Z, obs, fold, cot=None, ck_from="train"):
    """Control references for the held-out donors of one CV fold.

    c_k: PBS cells of cell type k from the training donors (ck_from="train", as in training) or from the
         held-out donors (ck_from="query").
    c_{d,k}: the held-out donor's own PBS cells (c_k if none).
    c_hat_{d,k}: CellOT counterfactual controls of that donor's cytokine cells of cell type k (c_k if none).
    Returns the held-out cytokine cells (rows) with small centroid tables and per-cell indices:
    ck[ct_idx], cdk[grp_idx], ccf[grp_idx].
    """
    test = obs["donor"].isin(FOLD_DONORS[fold]).to_numpy()
    pbs = (obs["cytokine"] == "PBS").to_numpy()
    cts = np.unique(obs["cell_type"])
    ct = obs["cell_type"].to_numpy()
    donor = obs["donor"].to_numpy()

    def mean_or(mask, fallback):
        idx = np.flatnonzero(mask)
        return Z[idx].mean(axis=0) if len(idx) else fallback

    zero = np.zeros(Z.shape[1], dtype=np.float32)
    ref_donors = test if ck_from == "query" else ~test
    ck = {c: mean_or((ct == c) & pbs & ref_donors, zero) for c in cts}
    cdk = {(d, c): mean_or((donor == d) & (ct == c) & pbs, ck[c]) for d in FOLD_DONORS[fold] for c in cts}
    rows = np.flatnonzero(~pbs & test)
    cf = {}
    if cot is not None:
        cf = counterfactual_centroids(cot, Z, rows, pd.MultiIndex.from_arrays([donor[rows], ct[rows]]))
    groups = list(cdk)
    gpos = {g: i for i, g in enumerate(groups)}
    cpos = {c: i for i, c in enumerate(cts)}
    return {
        "rows": rows,
        "ct_idx": np.array([cpos[c] for c in ct[rows]]),
        "grp_idx": np.array([gpos[k] for k in zip(donor[rows], ct[rows])]),
        "ck": np.stack([ck[c] for c in cts]),
        "cdk": np.stack([cdk[g] for g in groups]),
        "ccf": np.stack([cf.get(g, ck[g[1]]) for g in groups]) if cot is not None else None,
    }


def macro_auroc(y, probs, n_classes):
    """Mean one-vs-rest AUROC over the classes present in y."""
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y == c, probs[:, c]) for c in range(n_classes) if (y == c).any() and not (y == c).all()]
    return float(np.mean(aucs)) if aucs else 0.0


def load_classifier(stem):
    """CV model saved as {stem}.pt (cytap format) or {stem}.pth (state_dict)."""
    import torch
    from cytap.model import CytokineClassifier, _strip_compile_prefix, load_model
    stem = Path(stem)
    if stem.with_suffix(".pt").exists():
        return load_model(str(stem.with_suffix(".pt")))
    sd = _strip_compile_prefix(torch.load(stem.with_suffix(".pth"), map_location="cpu", weights_only=True))
    m = CytokineClassifier(sd["network.0.weight"].shape[1], sd["network.8.weight"].shape[0])
    m.load_state_dict(sd)
    return m.to("cuda" if torch.cuda.is_available() else "cpu").eval()


# ---------------------------------------------------------------- benchmark helpers
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
         "κ": "kappa", "λ": "lambda", "ω": "omega", "μ": "mu"}


def norm_cytokine(s) -> str:
    """Cytokine name for matching across resources: Greek letters spelled out, lower case, no punctuation."""
    s = str(s)
    for g, a in GREEK.items():
        s = s.replace(g, a)
    return s.lower().replace("-", "").replace("/", "").replace(" ", "").replace("_", "")
