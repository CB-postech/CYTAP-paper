"""CellOT counterfactual controls: inputs for CellOT training, and conversion of the trained maps for cytap.

prepare  State embeddings reduced to 100 principal components (input of train_cellot.sh).
         Parse-PBMC: one dataset per CV fold (the three held-out donors) and one with all donors ("nofold");
         condition column 'treatment' (cytokine / PBS). --target train: the held-out donors' cytokine cells
         and the training donors' PBS cells (parse_runs_trainctrl).
         COMBAT: HV, COVID_MILD, COVID_SEV and COVID_CRIT cells of data.combat; condition column 'Source'.
         -> {cellot.datasets}/{run}_data_pca100.h5ad (COMBAT: combat_pca100.h5ad) and {run}_pca_model_100d.pkl
export   trained CellOT maps (config.yaml + cache/model.pt) and their PCA model -> one cytap CellOT file per run,
         {cellot.maps}/{run}_cellot.pt, used by cytap.predict (02_inference), eval_control_references.py and
         07_variance_partitioning (cytap environment; CellOT itself is not needed).

python 05_cellot/cellot_maps.py prepare parse [--runs 1 2 3 4 nofold]
python 05_cellot/cellot_maps.py prepare parse --target train --runs 1 2 3 4
python 05_cellot/cellot_maps.py prepare combat
python 05_cellot/cellot_maps.py export parse
python 05_cellot/cellot_maps.py export parse --target train --runs 1 2 3 4
python 05_cellot/cellot_maps.py export combat

The published maps are on Zenodo (results/cellot/). Their PCA models were fitted before random_state was fixed,
so retraining gives similar but not identical maps.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np

from paper_utils import CFG, FOLD_DONORS, path

N_PCS = 100
COMBAT_SOURCES = ["HV", "COVID_MILD", "COVID_SEV", "COVID_CRIT"]


def run_names(dataset, runs, target="heldout"):
    if dataset == "combat":
        return {"combat": None}
    if target == "train":
        assert "nofold" not in runs, "--target train is defined for the CV folds only"
    key = "parse_runs_trainctrl" if target == "train" else "parse_runs"
    return {CFG["cellot"][key][r if r == "nofold" else int(r)]: r for r in runs}


def prepare(dataset, runs, target="heldout"):
    from sklearn.decomposition import PCA
    d = path("cellot", "datasets")
    d.mkdir(parents=True, exist_ok=True)

    def write(X, obs, name, data_name):
        if (d / f"{name}_pca_model_100d.pkl").exists():
            print(f"skip (exists): {d / f'{name}_pca_model_100d.pkl'}")
            return
        pca = PCA(n_components=N_PCS, random_state=0)
        Xp = pca.fit_transform(X)
        print(f"{name}: {X.shape} -> {Xp.shape}, {100 * pca.explained_variance_ratio_.sum():.2f}% variance")
        with open(d / f"{name}_pca_model_100d.pkl", "wb") as f:
            pickle.dump(pca, f)
        ad.AnnData(X=Xp.astype(np.float32), obs=obs.copy()).write_h5ad(d / data_name)

    if dataset == "parse":
        src = ad.read_h5ad(path("parse", "emb"), backed="r")
        obs = src.obs[["treatment", "cytokine", "donor", "cell_type"]].copy()
        Z = np.asarray(src.obsm["X_state"][:], dtype=np.float32)
        pbs = (obs["treatment"] == "PBS").to_numpy()
        for name, run in run_names("parse", runs, target).items():
            if run == "nofold":
                keep = np.ones(len(obs), bool)
            else:
                held = obs["donor"].isin(FOLD_DONORS[int(run)]).to_numpy()
                keep = (held & ~pbs) | (~held & pbs) if target == "train" else held
            write(Z[keep], obs[keep], name, f"{name}_data_pca100.h5ad")
    else:
        src = ad.read_h5ad(path("data", "combat"))
        src = src[src.obs["Source"].isin(COMBAT_SOURCES)]
        obs = src.obs[["Source", "donor_id", "major_subset"]].copy()
        obs["Source"] = obs["Source"].astype(str)
        write(np.asarray(src.obsm["X_state"], dtype=np.float32), obs, "combat", "combat_pca100.h5ad")


def export(dataset, runs, target="heldout"):
    from cytap.counterfactual import CellOT
    d, res = path("cellot", "datasets"), path("cellot", "results")
    for name in run_names(dataset, runs, target):
        f = path("cellot", "maps") / f"{name}_cellot.pt"
        if f.exists():
            print(f"skip (exists): {f}")
            continue
        if dataset == "combat":
            dirs = {s: res / r for s, r in CFG["cellot"]["combat_runs"].items()}
            cot = CellOT.from_cellot(dirs, d / "combat_pca_model_100d.pkl", condition_key="Source", control_label="HV")
        else:
            cot = CellOT.from_cellot({"cytokine": res / name / "cyt_to_PBS_pc100"}, d / f"{name}_pca_model_100d.pkl",
                                     condition_key="treatment", control_label="PBS")
        f.parent.mkdir(parents=True, exist_ok=True)
        cot.save(f)
        print(f"{name}: maps {cot.source_labels}, best step {[v['best_step'] for v in cot.info.values()]} -> {f}")


p = argparse.ArgumentParser()
p.add_argument("step", choices=["prepare", "export"])
p.add_argument("dataset", choices=["parse", "combat"])
p.add_argument("--runs", nargs="+", default=["1", "2", "3", "4", "nofold"])
p.add_argument("--target", choices=["heldout", "train"], default="heldout",
               help="Parse-PBMC folds: PBS cells of the held-out donors or of the training donors")
a = p.parse_args()
(prepare if a.step == "prepare" else export)(a.dataset, a.runs, a.target)
