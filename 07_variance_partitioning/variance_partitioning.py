"""Five-component variance partitioning of the input representations (sum of squares), Parse-PBMC.

Centroids: each representation is summarised as centroids Y (one row per donor x cell type x cytokine, cytokine-
treated cells of all 12 donors). c_k: PBS cells of cell type k; c_dk: donor d's own PBS cells (c_k if none);
c_hat_dk: mean CellOT counterfactual control of the donor's cytokine cells (map trained on all donors,
cellot.parse_runs.nofold; c_k if none). All cells of a group share these references, so the centroid of a
representation is the same transform applied to the group's mean embedding.
  --set inputs : Baseline, Donor_Only, Sub, Donor_Sub (= Raw, Adj, Dev, Comp)
  --set cellot : Raw and Adj / Dev / Comp with true, pooled (c_dk := c_k) and CellOT references

Partitioning: Y is regressed on the full-factorial, sum-coded design cell_type * treatment + donor * cell_type.
Components = term blocks (Cell_type, Donor, Donor_x_Cell_type, Treatment, Treatment_x_Cell_type); independent
contributions are the hierarchical-partitioning (Shapley) averages of the explained SS over all 2^5 component
subsets; Residual = TSS - SS(full model). Reported in embedding units (SS_abs) and as % of the representation's
own TSS. SS is additive over feature blocks, so
SS(Comp) = SS(Adj) + SS(Dev) is checked as a consistency test.

Output: results/variance_partitioning/{set}/vp5_hierarchical_SS.csv,
        vp5_permutation_null.csv (--n-perm: cell-type labels permuted within donor x treatment),
        centroids/centroid_{Condition}.npz (reused when present), cellot/counterfactual_diagnostics.csv

python 07_variance_partitioning/variance_partitioning.py --set inputs [--n-perm 20]
python 07_variance_partitioning/variance_partitioning.py --set cellot
"""
import argparse
import gc
import itertools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import patsy
from scipy.linalg import lstsq

from cytap.features import group_means
from paper_utils import cellot_map, compose_input, counterfactual_centroids, load_parse, out

SETS = {
    "inputs": {"Baseline": ("raw", "cdk"), "Donor_Only": ("adj", "cdk"), "Sub": ("dev", "cdk"),
               "Donor_Sub": ("comp", "cdk")},
    "cellot": {"Raw": ("raw", "cdk"), "Adj_true": ("adj", "cdk"), "Dev_true": ("dev", "cdk"),
               "Comp_true": ("comp", "cdk"), "Dev_pool": ("dev", "ck"), "Comp_pool": ("comp", "ck"),
               "Adj_cf": ("adj", "ccf"), "Dev_cf": ("dev", "ccf"), "Comp_cf": ("comp", "ccf")},
}
# (composite, first block, second block): SS(composite) must equal the sum of the blocks.
# Pooled reference: c_dk = c_k, so the first block of Comp_pool is the raw embedding.
ADDITIVE = [("Donor_Sub", "Donor_Only", "Sub"), ("Comp_true", "Adj_true", "Dev_true"),
            ("Comp_pool", "Raw", "Dev_pool"), ("Comp_cf", "Adj_cf", "Dev_cf")]

FULL_FORMULA = "C(cell_type, Sum)*C(treatment, Sum) + C(donor, Sum)*C(cell_type, Sum)"
COMP_FACTORS = {
    "Cell_type": frozenset({"C(cell_type, Sum)"}),
    "Donor": frozenset({"C(donor, Sum)"}),
    "Donor_x_Cell_type": frozenset({"C(donor, Sum)", "C(cell_type, Sum)"}),
    "Treatment": frozenset({"C(treatment, Sum)"}),
    "Treatment_x_Cell_type": frozenset({"C(treatment, Sum)", "C(cell_type, Sum)"}),
}
COMPONENTS = list(COMP_FACTORS)


def build_design(meta: pd.DataFrame):
    """Design matrix, intercept columns and the column block of each component."""
    dm = patsy.dmatrix(FULL_FORMULA, meta, return_type="matrix")
    key2cols = {frozenset(f.name() for f in term.factors): np.arange(sl.start, sl.stop)
                for term, sl in dm.design_info.term_slices.items()}
    missing = [c for c in COMPONENTS if COMP_FACTORS[c] not in key2cols]
    if missing:
        raise RuntimeError(f"terms not found in design: {missing} ({dm.design_info.term_names})")
    D = np.ascontiguousarray(np.asarray(dm, dtype=np.float64))
    return D, key2cols[frozenset()], {c: key2cols[COMP_FACTORS[c]] for c in COMPONENTS}


class SSSolver:
    """RSS of sub-models from one Gram matrix: RSS(S) = ||Y||^2 - <beta_S, D_S'Y>."""

    def __init__(self, D, Y):
        Y = np.asarray(Y, dtype=np.float64)
        self.G, self.B = D.T @ D, D.T @ Y
        self.ssY = float(np.einsum("ij,ij->", Y, Y))

    def rss(self, cols):
        Bs = self.B[cols]
        beta = lstsq(self.G[np.ix_(cols, cols)], Bs, lapack_driver="gelsy", check_finite=False)[0]
        return self.ssY - float(np.einsum("ij,ij->", beta, Bs))


def cols_for(subset, intercept, comp_cols):
    return np.unique(np.concatenate([intercept] + [comp_cols[c] for c in subset]))


def partition(Y, meta, label):
    D, intercept, comp_cols = build_design(meta)
    t0 = time.time()
    solver = SSSolver(D, Y)
    tss = solver.rss(intercept)
    ss_sub = {frozenset(s): tss - solver.rss(cols_for(s, intercept, comp_cols))
              for r in range(len(COMPONENTS) + 1) for s in itertools.combinations(COMPONENTS, r)}

    n = len(COMPONENTS)
    contrib = {}
    for c in COMPONENTS:
        others = [x for x in COMPONENTS if x != c]
        contrib[c] = sum(
            math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
            * (ss_sub[frozenset(S + (c,))] - ss_sub[frozenset(S)])
            for r in range(n) for S in itertools.combinations(others, r))

    ss_full = ss_sub[frozenset(COMPONENTS)]
    residual = tss - ss_full
    hp = [{"Condition": label, "Component": c, "SS_abs": contrib[c], "Pct_of_TSS": 100 * contrib[c] / tss}
          for c in COMPONENTS]
    hp.append({"Condition": label, "Component": "Residual", "SS_abs": residual, "Pct_of_TSS": 100 * residual / tss})

    print(f"  {label}: design {D.shape}, Y {Y.shape}, {time.time() - t0:.0f}s, explained {100 * ss_full / tss:.1f}%")
    return hp


def permute_celltype(meta, rng):
    """Shuffle cell-type labels within each donor x treatment group."""
    out_meta = meta.copy()
    ct = out_meta["cell_type"].to_numpy().copy()
    for idx in out_meta.groupby(["donor", "treatment"], sort=False).indices.values():
        ct[idx] = rng.permutation(ct[idx])
    out_meta["cell_type"] = ct
    return out_meta


def check_additivity(df_hp):
    piv = df_hp.pivot(index="Condition", columns="Component", values="SS_abs")
    for comp, a, b in ADDITIVE:
        if {comp, a, b} <= set(piv.index):
            rel = ((piv.loc[comp] - piv.loc[a] - piv.loc[b]).abs() / piv.loc[comp].abs().clip(lower=1e-12)).max()
            print(f"additivity SS({comp}) = SS({a}) + SS({b}): max rel. error {rel:.1e}"
                  + ("  <-- check centroid construction" if rel > 1e-6 else ""))


# ---------------------------------------------------------------- centroids
def build_centroids(set_name, cdir):
    """{Condition: (Y, meta)} for the representations of the set (see module docstring)."""
    Z, obs = load_parse()
    pbs = (obs["cytokine"] == "PBS").to_numpy()
    classes = np.unique(obs["cytokine"].to_numpy()[~pbs])
    keys = pd.DataFrame({"donor": obs["donor"], "cell_type": obs["cell_type"],
                         "treatment": "T_" + pd.Series(np.searchsorted(classes, obs["cytokine"])).astype(str)})
    rows = np.flatnonzero(~pbs)
    idx = pd.MultiIndex.from_frame(keys.iloc[rows].reset_index(drop=True))
    codes, uniq = pd.factorize(idx, sort=True)
    sub = np.full(len(Z), -1, dtype=np.int64)
    sub[rows] = codes
    mz, _ = group_means(Z, sub, sub >= 0, len(uniq))        # mean embedding per donor x cell type x treatment
    meta = pd.DataFrame(list(uniq), columns=["donor", "cell_type", "treatment"])
    pairs = list(zip(meta["donor"], meta["cell_type"]))

    # references (float64)
    ct_p, d_p, Zp = obs["cell_type"].to_numpy()[pbs], obs["donor"].to_numpy()[pbs], Z[pbs].astype(np.float64)
    c_codes, cts = pd.factorize(ct_p)
    ck_tab, _ = group_means(Zp, c_codes, np.ones(len(Zp), bool), len(cts))
    c_k = dict(zip(cts, ck_tab))
    g_codes, groups = pd.factorize(pd.MultiIndex.from_arrays([d_p, ct_p]))
    cdk_tab, _ = group_means(Zp, g_codes, np.ones(len(Zp), bool), len(groups))
    c_dk = dict(zip(groups, cdk_tab))
    del Zp
    tables = {"ck": np.array([c_k[k] for _, k in pairs]), "cdk": np.array([c_dk.get(g, c_k[g[1]]) for g in pairs])}

    if set_name == "cellot":
        ch_dk = counterfactual_centroids(cellot_map("nofold"), Z, rows, pd.MultiIndex.from_arrays(
            [obs["donor"].to_numpy()[rows], obs["cell_type"].to_numpy()[rows]]), dtype="float64")
        tables["ccf"] = np.array([ch_dk.get(g, c_k[g[1]]) for g in pairs])
        diag = []   # distance of the counterfactual centroids to the true ones, relative to c_k
        for (d, k), true in c_dk.items():
            if (d, k) in ch_dk:
                dt, dh = true - c_k[k], ch_dk[(d, k)] - c_k[k]
                diag.append({"donor": d, "cell_type": k, "err_cf": np.linalg.norm(ch_dk[(d, k)] - true),
                             "err_pool": np.linalg.norm(dt), "shrink": np.linalg.norm(dh) / (np.linalg.norm(dt) + 1e-12),
                             "cos_dir": float(dh @ dt / (np.linalg.norm(dh) * np.linalg.norm(dt) + 1e-12))})
        diag = pd.DataFrame(diag).assign(R=lambda x: x.err_cf / x.err_pool)
        diag.to_csv(out("variance_partitioning", "cellot", "counterfactual_diagnostics.csv"), index=False)
        print(diag[["err_cf", "err_pool", "R", "shrink", "cos_dir"]].median().to_string())
    del Z
    gc.collect()

    M = mz.astype(np.float64)
    res = {}
    for name, (rep, ref) in SETS[set_name].items():
        Y = compose_input(M, tables["ck"], tables[ref], rep)
        np.savez_compressed(cdir / f"centroid_{name}.npz", Y_raw=Y, donor=meta["donor"].values,
                            cell_type=meta["cell_type"].values, treatment=meta["treatment"].values)
        res[name] = (Y, meta)
    return res


def load_centroids(set_name, cdir):
    files = {n: cdir / f"centroid_{n}.npz" for n in SETS[set_name]}
    if not all(f.exists() for f in files.values()):
        return build_centroids(set_name, cdir)
    res = {}
    for n, f in files.items():
        with np.load(f, allow_pickle=True) as d:
            res[n] = (np.asarray(d["Y_raw"], dtype=np.float64),
                      pd.DataFrame({k: d[k].astype(str) for k in ["donor", "cell_type", "treatment"]}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True)
    ap.add_argument("--n-perm", type=int, default=0, help="cell-type label permutations (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cdir = out("variance_partitioning", args.set, "centroids", "x").parent
    hp_rows, perm_rows = [], []
    for cond, (Y, meta) in load_centroids(args.set, cdir).items():
        hp_rows += partition(Y, meta, cond)
        if args.n_perm:
            rng = np.random.default_rng(args.seed)  # same permutations for every condition
            for p in range(args.n_perm):
                php = partition(Y, permute_celltype(meta, rng), f"{cond}_perm{p}")
                rec = {"Condition": cond, "perm": p}
                for c in ["Cell_type", "Treatment_x_Cell_type"]:
                    rec[f"HP_{c}"] = next(x["SS_abs"] for x in php if x["Component"] == c)
                perm_rows.append(rec)

    df_hp = pd.DataFrame(hp_rows)
    df_hp.to_csv(out("variance_partitioning", args.set, "vp5_hierarchical_SS.csv"), index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(out("variance_partitioning", args.set, "vp5_permutation_null.csv"), index=False)
    print(df_hp.pivot(index="Condition", columns="Component", values="Pct_of_TSS").round(2).to_string())
    check_additivity(df_hp)


if __name__ == "__main__":
    main()
