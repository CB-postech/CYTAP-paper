"""ContactTracing on the Parse-PBMC benchmark (run inside the ContactTracing docker image).

Per responsive condition (cell type C, cytokine X): one MAST interaction test X vs PBS in C
(MAST-interactionTest.R of the ContactTracing repository; covariates cngeneson + donor) for the union
of the cognate receptors of all candidate cytokines (Parse-PBMC_cytokine_receptors.xlsx) expressed in
>= 2% of the cells. Response genes = HVGs of the cell type (cell_ranger, batch = donor) + receptors.
Receptor activity = number of response genes with interaction FDR < 0.05 (numSigI1); cytokine score =
max over its receptors; candidates ranked by score (ties ranked conservatively); true_rank = rank of X.
Cells are downsampled to <= 200 per donor x cell type x condition (seed 0); conditions with < 30 cells
in either group are skipped.

Output (ContactTracing_output/): downsampled/*.h5ad, interaction_test/, run_log.csv,
contacttracing_true_rank.csv, ct_pool_by_condition.csv (pool = cytokines with a tested receptor)

  docker run ... -v $PWD:/data <ContactTracing image>      # repository root at /data (and any symlinked data folders)
  cd /data
  python 09_benchmark_parse/contacttracing.py downsample
  python 09_benchmark_parse/contacttracing.py run --njobs 50 [--celltypes "CD14 Mono" ...]
  python 09_benchmark_parse/contacttracing.py rank
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import scale

from paper_utils import GREEK, bench, norm_cytokine, path

CONTROL, CAP, MIN_CELLS, FRAC_EXP, SEED, JOB_TIMEOUT = "PBS", 200, 30, 0.02, 0, 1800
OUT = bench("parse_dir", "ContactTracing_output", "x").parent
DOWN = OUT / "downsampled"
TUTORIAL = path("benchmark", "contacttracing_tutorial")

responsive = pd.read_csv(path("benchmark", "parse_dir") / "benchmark" / "responsive_conditions.csv")
celltypes = sorted(responsive["celltype"].unique())


def safe_label(s):
    """R-formula / path safe: Greek -> ASCII, other characters -> _, no leading digit."""
    s = str(s)
    for g, a in GREEK.items():
        s = s.replace(g, a)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    return "X" + s if re.match(r"^[0-9]", s) else s


def safe_ct(c):
    return c.replace("/", "_").replace(" ", "_")


cr = pd.read_excel(path("reference", "cytokine_receptors"))
xls = {norm_cytokine(r["Cytokines"]): ([] if pd.isna(r["Receptor gene symbol"]) else
                                  [x.strip() for x in re.split(r"[;,]", str(r["Receptor gene symbol"])) if x.strip()])
       for _, r in cr.iterrows()}
CYT2REC = {c: xls[norm_cytokine(c)] for c in sorted(responsive["cytokine"].unique()) if norm_cytokine(c) in xls}
ALL_RECEPTORS = sorted({g for recs in CYT2REC.values() for g in recs})
CYTO_SAFE = {c: safe_label(c) for c in responsive["cytokine"].unique()}
CTRL_SAFE = safe_label(CONTROL)
assert len(set(CYTO_SAFE.values())) == len(CYTO_SAFE), "safe_label collision"


# ---------------------------------------------------------------- downsampling
def downsample():
    a = sc.read_h5ad(path("parse", "counts"), backed="r")
    obs = a.obs[["donor", "cell_type", "cytokine"]].copy()
    rng = np.random.default_rng(SEED)
    DOWN.mkdir(parents=True, exist_ok=True)
    for c in celltypes:
        cyts = set(responsive.loc[responsive["celltype"] == c, "cytokine"]) | {CONTROL}
        sel = obs[(obs["cell_type"] == c) & obs["cytokine"].isin(cyts)]
        keep = []
        for _, idx in sel.groupby(["donor", "cytokine"]).groups.items():
            idx = np.asarray(idx)
            keep.append(rng.choice(idx, CAP, replace=False) if len(idx) > CAP else idx)
        sub = a[np.concatenate(keep)].to_memory()
        sub.layers["counts"] = sub.X.copy()
        sub.write(DOWN / f"{safe_ct(c)}.h5ad")
        print(f"{c}: {sub.n_obs} cells | PBS {int((sub.obs.cytokine == CONTROL).sum())}", flush=True)


# ---------------------------------------------------------------- interaction tests
def prep_celltype(c):
    a = sc.read_h5ad(DOWN / f"{safe_ct(c)}.h5ad")
    a.obs["cell type"] = a.obs["cell_type"].astype(str)
    a.obs["condition"] = a.obs["cytokine"].astype(str)
    a.obs["sample"] = a.obs["donor"].astype(str)
    a.obs["donor"] = a.obs["donor"].astype(str)
    tmp = anndata.AnnData(a.layers["counts"].copy(), obs=a.obs[[]].copy(), var=a.var[[]].copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    a.layers["X"] = tmp.X
    lx = tmp.X.tocsr().astype(np.float32) if issparse(tmp.X) else np.asarray(tmp.X, np.float32)
    if issparse(lx):
        lx.data = np.log10(lx.data + 0.1) - np.log10(0.1)
    else:
        lx = np.log10(lx + 0.1) - np.log10(0.1)
    a.layers["logX"] = lx                 # log10(x + 0.1) - log10(0.1), as in the ContactTracing tutorial
    a.layers.pop("counts")
    try:
        sc.pp.highly_variable_genes(a, flavor="cell_ranger", batch_key="sample", n_bins=20, max_mean=np.inf, layer="X")
    except Exception as e:
        print("  batched HVG failed, no batch:", e)
        sc.pp.highly_variable_genes(a, flavor="cell_ranger", n_bins=20, max_mean=np.inf, layer="X")
    a.layers.pop("X")
    hvg = set(a.var.index[a.var["highly_variable"]])
    a.uns["keepGenes"] = sorted(hvg | set(ALL_RECEPTORS))
    a.obs["cngeneson"] = scale(np.asarray((a.layers["logX"] > 0).sum(axis=1)).flatten())
    print(f"{c}: {len(a.uns['keepGenes'])} response genes (HVG {len(hvg)} + receptors)", flush=True)
    return a


def write_commands(a, c, cyt, intdir, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(intdir, exist_ok=True)
    cs = CYTO_SAFE[cyt]
    b = a[a.obs["condition"].isin([cyt, CONTROL]), [g for g in a.uns["keepGenes"] if g in a.var.index]].copy()
    ns, nc = int((b.obs["condition"] == cyt).sum()), int((b.obs["condition"] == CONTROL).sum())
    if ns < MIN_CELLS or nc < MIN_CELLS:
        return None, set(), 0, (ns, nc)
    b.obs["condition"] = b.obs["condition"].map({cyt: cs, CONTROL: CTRL_SAFE}).astype(str)
    if issparse(b.layers["logX"]):
        b.layers["logX"] = b.layers["logX"].toarray()
    b.var["fracExp"] = (b.layers["logX"] > 0).sum(axis=0) / b.shape[0]
    recs = [r for r in ALL_RECEPTORS if r in b.var.index and b.var.loc[r, "fracExp"] >= FRAC_EXP]
    if not recs:
        return None, set(), 0, (ns, nc)
    infile = f"{tmpdir}/{safe_ct(c)}__{cs}.h5ad"
    b.write(infile)
    logX = b.layers["logX"]
    f1, f2 = np.asarray(b.obs["condition"] == cs), np.asarray(b.obs["condition"] == CTRL_SAFE)
    cmd_path, expected, n = f"{intdir}/commands.sh", set(), 0
    with open(cmd_path, "w") as fh:
        for g in recs:
            of = f"{intdir}/{g}/interaction_{cs}_vs_{CTRL_SAFE}.txt"
            fe = np.asarray(logX[:, np.where(b.var.index == g)[0]] > 0).flatten()
            # >= 2 expressing and >= 2 non-expressing cells, both present in each condition
            if fe.sum() < 2 or (~fe).sum() < 2 or not ((fe & f1).any() and (fe & f2).any()
                                                       and (~fe & f1).any() and (~fe & f2).any()):
                continue
            expected.add(of)
            if os.path.exists(of):
                continue
            fh.write(f'mkdir -p "{intdir}/{g}" && timeout {JOB_TIMEOUT} env MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
                     f'OMP_NUM_THREADS=1 R --vanilla --args --infile "{infile}" --outdir "{intdir}" '
                     f'--conditions "{cs}" "{CTRL_SAFE}" --covariates cngeneson,donor --numcore 1 --genes "{g}"'
                     f' < {TUTORIAL}/R/MAST-interactionTest.R &> "{intdir}/{g}/stdout.txt"\n')
            n += 1
    return cmd_path, expected, n, (ns, nc)


def run(njobs, only):
    sys.path.append(str(TUTORIAL / "python"))
    import contactTracing_library as ct
    log = []
    for c in only or celltypes:
        a = prep_celltype(c)
        for cyt in sorted(responsive.loc[responsive["celltype"] == c, "cytokine"]):
            intdir = f"{OUT}/interaction_test/{safe_ct(c)}/{CYTO_SAFE[cyt]}"
            cmd_path, expected, n, (ns, nc) = write_commands(a, c, cyt, intdir, f"{OUT}/tmp/{safe_ct(c)}")
            if cmd_path is None:
                log.append((c, cyt, f"skip ns={ns} nc={nc}"))
                continue
            if n > 0:
                ct.run_in_background(f"parallel -j {njobs} < {cmd_path}", stdoutfile=f"{intdir}/commands_stdout.txt",
                                     wait=True)
            done = sum(os.path.exists(f) for f in expected)
            log.append((c, cyt, f"ran {done}/{len(expected)}"))
            print(f"  {cyt}: {done}/{len(expected)}", flush=True)
        pd.DataFrame(log, columns=["celltype", "cytokine", "status"]).to_csv(OUT / "run_log.csv", index=False)


# ---------------------------------------------------------------- ranking
def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    fdr = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.clip(fdr, 0, 1)
    return out


def numsig(f, alpha=0.05):
    try:
        df = pd.read_csv(f, sep="\t")
    except Exception:
        return None
    if "pval" not in df.columns:
        return None
    p = pd.to_numeric(df["pval"], errors="coerce").dropna().values
    return int((bh(p) < alpha).sum()) if len(p) else 0


def rank():
    rows, pools = [], []
    for c in celltypes:
        for cyt in sorted(responsive.loc[responsive["celltype"] == c, "cytokine"]):
            cs = CYTO_SAFE[cyt]
            run_dir = f"{OUT}/interaction_test/{safe_ct(c)}/{cs}"
            if not os.path.isdir(run_dir):
                rows.append((c, cyt, np.nan))
                continue
            rec = {}
            for g in os.listdir(run_dir):
                f = f"{run_dir}/{g}/interaction_{cs}_vs_{CTRL_SAFE}.txt"
                if os.path.exists(f) and (v := numsig(f)) is not None:
                    rec[g] = v
            score = {cand: max(rec[g] for g in recs if g in rec) for cand, recs in CYT2REC.items()
                     if any(g in rec for g in recs)}
            r = int(sum(v > score[cyt] for v in score.values())) + 1 if cyt in score else np.nan
            rows.append((c, cyt, r))
            if score:
                pools.append((c, cyt, len(score), r))
    tr = pd.DataFrame(rows, columns=["celltype", "cytokine", "true_rank"])
    tr.to_csv(OUT / "contacttracing_true_rank.csv", index=False)
    pool = pd.DataFrame(pools, columns=["celltype", "cytokine", "pool_size", "true_rank"])
    pool.to_csv(OUT / "ct_pool_by_condition.csv", index=False)
    print(f"evaluable: {tr['true_rank'].notna().sum()}/{len(tr)} | pool size mean {pool['pool_size'].mean():.1f} "
          f"(min {pool['pool_size'].min()}, max {pool['pool_size'].max()})")


p = argparse.ArgumentParser()
p.add_argument("step", choices=["downsample", "run", "rank"])
p.add_argument("--njobs", type=int, default=50)
p.add_argument("--celltypes", nargs="+")
a = p.parse_args()
print(f"{len(CYT2REC)}/{responsive['cytokine'].nunique()} cytokines with receptors | {len(ALL_RECEPTORS)} receptors")
if a.step == "downsample":
    downsample()
elif a.step == "run":
    run(a.njobs, a.celltypes)
else:
    rank()
