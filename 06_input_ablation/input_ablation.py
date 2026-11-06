"""Input ablation: donor-held-out 4-fold cross-validation on Parse-PBMC, and the full pipeline on the macrophage data.

cv          Inputs (names in the result table): raw (Baseline) z; adj (Donor_Only) z - (c_dk - c_k); dev (Sub)
            z - c_dk; comp (Donor_Sub) [adj ; dev], the CYTAP input; hvg (HVG_Baseline) log-normalized expression of
            2,058 HVGs (00_preprocessing/preprocess.py parse-hvg). c_k: PBS centroid of cell type k over the
            training donors only; c_dk: each donor's own PBS centroid (training or held-out donor), c_k if none.
            Classifier and training settings of the CYTAP model (cytap.train) with drop_last=True (HVG: training
            cells shuffled once, seed 42). Test macro F1 and macro one-vs-rest AUROC (overall and per cell type)
            after every epoch -> results/input_ablation/all_ablation_results.csv and models/{Input}_Fold{k}.pt (the
            published models, used by 05_cellot and 09_benchmark_parse, are in cv_models of config/paths.yaml)
macrophage  The full-training models of each input (01_training) applied to the macrophage data (paired) and
            analysed as the CYTAP result in 03_query_analysis: IFNb+ vs Control and the within-donor IFN-beta logit
            change. HVG model: expression of the training HVGs (missing genes = 0).
            -> results/input_ablation/Mac_IFNb_top_cytokine_input_{raw,adj,dev,HVG}.csv, Mac_IFN_beta_delta_input_*.csv

python 06_input_ablation/input_ablation.py cv --inputs raw adj dev comp
python 06_input_ablation/input_ablation.py cv --inputs hvg
python 06_input_ablation/input_ablation.py macrophage
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

import cytap
from cytap.features import compute_centroids, group_means
from cytap.model import save_model
from cytap.predict import predict_proba
from cytap.train import train_classifier
from paper_utils import (FOLD_DONORS, TABLE_NAMES, compare_table, compose_input, delta_table, load_hvg, load_parse,
                         macro_auroc, out, path, predict_input)

CHUNK = 1_000_000


# ---------------------------------------------------------------- cross-validation
def make_evaluator(X_test, y_test, ct_test, n_classes, gamma=2.0):
    def evaluate(model, epoch):
        probs = predict_proba(model, X_test)
        pred = probs.argmax(1)
        p_true = np.clip(probs[np.arange(len(y_test)), y_test], 1e-12, 1)
        top5 = np.argsort(-probs, axis=1)[:, :5]
        m = {
            "test_loss": float(np.mean((1 - p_true) ** gamma * -np.log(p_true))),  # focal loss
            "test_accuracy": 100 * float((pred == y_test).mean()),
            "test_accuracy_top5": 100 * float((top5 == y_test[:, None]).any(1).mean()),
            "Overall/Macro_F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "Overall/AUROC": macro_auroc(y_test, probs, n_classes),
        }
        for ct in np.unique(ct_test):
            s = ct_test == ct
            key = str(ct).replace(" ", "_")
            m[f"CellType_Macro_F1/{key}"] = f1_score(y_test[s], pred[s], average="macro", zero_division=0)
            m[f"CellType_AUROC/{key}"] = macro_auroc(y_test[s], probs[s], n_classes)
        print(f"  epoch {epoch}: test macro F1 {m['Overall/Macro_F1']:.4f} | AUROC {m['Overall/AUROC']:.4f}")
        return m
    return evaluate


def cv(inputs, no_compile=False):
    table = out("input_ablation", "all_ablation_results.csv")
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    model_dir = out("input_ablation", "models", "x").parent

    def run_fold(name, fold, X_tr, y_tr, X_te, y_te, ct_te, classes, preshuffle=False):
        f = model_dir / f"{name}_Fold{fold}.pt"
        if f.exists():
            print(f"skip {name} fold {fold} (model exists)")
            return
        print(f"\n=== {name} fold {fold}: train {X_tr.shape} | test {X_te.shape}")
        history = []
        model = train_classifier(X_tr, y_tr, classes, preshuffle=preshuffle, drop_last=True, log_fn=history.append,
                                 compile=not no_compile, num_workers=0 if no_compile else 4,
                                 epoch_callback=make_evaluator(X_te, y_te, ct_te, len(classes)))
        save_model(f, model, classes, {"representation": name, "fold": fold})
        rows.extend({"Condition": name, "Fold": fold, **h} for h in history)
        pd.DataFrame(rows).to_csv(table, index=False)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    emb = [r for r in inputs if r != "hvg"]
    if emb:
        Z, obs = load_parse()
        pbs = (obs["cytokine"] == "PBS").to_numpy()
        classes, y_cyt = np.unique(obs["cytokine"].to_numpy()[~pbs], return_inverse=True)
        y = np.full(len(obs), -1, dtype=np.int64)
        y[~pbs] = y_cyt
        ct_codes, ct_names = pd.factorize(obs["cell_type"])
        d_codes, d_names = pd.factorize(obs["donor"])
        nct = len(ct_names)
        g_codes = d_codes * nct + ct_codes

        def build(idx, ck, cdk, rep):   # input matrix in chunks to bound memory
            X = np.empty((len(idx), Z.shape[1] * (2 if rep == "comp" else 1)), dtype=np.float32)
            for i in range(0, len(idx), CHUNK):
                r = idx[i:i + CHUNK]
                X[i:i + len(r)] = compose_input(Z[r], ck[ct_codes[r]], cdk[g_codes[r]], rep)
            return X

        for fold, test_donors in FOLD_DONORS.items():
            test = obs["donor"].isin(test_donors).to_numpy()
            ck, _ = group_means(Z, ct_codes, pbs & ~test, nct)               # training donors only
            cdk, has = group_means(Z, g_codes, pbs, len(d_names) * nct)      # each donor's own PBS cells
            cdk[~has] = ck[np.arange(len(cdk))[~has] % nct]
            tr, te = np.flatnonzero(~pbs & ~test), np.flatnonzero(~pbs & test)
            for rep in emb:
                run_fold(TABLE_NAMES[rep], fold, build(tr, ck, cdk, rep), y[tr], build(te, ck, cdk, rep), y[te],
                         obs["cell_type"].to_numpy()[te], classes)
                gc.collect()
        del Z

    if "hvg" in inputs:
        X, y, donor, ct, classes = load_hvg()
        for fold, test_donors in FOLD_DONORS.items():
            test = np.isin(donor, test_donors)
            run_fold(TABLE_NAMES["hvg"], fold, X[~test], y[~test], X[test], y[test], ct[test], classes, preshuffle=True)


# ---------------------------------------------------------------- macrophage pipeline test
def macrophage():
    COND, CTRL, DONOR = "condition", "Control", "sample"
    TARGETS = ["IFNb+", "IFNg+", "IFNg+ fibro+", "IL4+", "TNFa+", "TNFa+ IFNg+", "TNFa+ IFNg+ fibro+", "TNFa+ fibro+"]

    def predict_hvg(adata, model):
        """Same cell filtering as cytap.predict(design="paired"), expression input instead of the embedding."""
        keep = compute_centroids(adata.obsm["X_state"], (adata.obs[COND] == CTRL).to_numpy(),
                                 donor=adata.obs[DONOR].to_numpy()).keep
        a = adata[keep].copy()
        genes = model.config["genes"]
        pos = a.var_names.get_indexer(genes)
        print(f"HVG model: {(pos >= 0).sum()}/{len(genes)} training genes found")
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        Xg = np.zeros((a.n_obs, len(genes)), dtype=np.float32)
        Xg[:, pos >= 0] = X[:, pos[pos >= 0]]
        a.obsm["pred_probs"], a.uns["pred_classes"] = cytap.predict_proba(model, Xg), np.asarray(model.classes)
        return a

    adata = ad.read_h5ad(path("data", "macrophage"))
    for rep, lab in {"raw": "raw", "adj": "adj", "dev": "dev", "hvg": "HVG"}.items():
        model = cytap.load_model(str(path("ablation_models", rep)))
        res = predict_hvg(adata, model) if rep == "hvg" else predict_input(adata, model, COND, CTRL, DONOR)
        res.write_h5ad(out("input_ablation", f"macrophage_cytap_{rep}.h5ad"))
        stats = cytap.stats.compare_conditions(res, COND, CTRL, "IFNb+", donor_key=DONOR, paired=True)
        compare_table(stats).to_csv(out("input_ablation", f"Mac_IFNb_top_cytokine_input_{lab}.csv"), index=False)
        dl = cytap.stats.paired_delta(res, "IFN-beta", COND, CTRL, TARGETS, DONOR)
        delta_table(dl).to_csv(out("input_ablation", f"Mac_IFN_beta_delta_input_{lab}.csv"), index=False)
        print(f"{rep}: {res.n_obs} cells")


p = argparse.ArgumentParser()
p.add_argument("step", choices=["cv", "macrophage"])
p.add_argument("--inputs", nargs="+", choices=list(TABLE_NAMES), default=["raw", "adj", "dev", "comp"])
p.add_argument("--no-compile", action="store_true", help="disable torch.compile (debugging)")
a = p.parse_args()
cv(a.inputs, a.no_compile) if a.step == "cv" else macrophage()
