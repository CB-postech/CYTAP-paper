"""Control references for unpaired inference, evaluated on Parse-PBMC held-out donors.

The 4-fold CV models (trained with true donor-matched centroids, 06_input_ablation) are applied
without retraining to the held-out donors' cytokine cells, with the control centroid replaced:

    Raw        z                          Baseline model
    Adj_true   z - (c_dk - c_k)           Donor_Only model
    Dev_true   z - c_dk                   Sub model
    Comp_true  [Adj_true ; Dev_true]      Donor_Sub model
    Dev_pool   z - c_k                    Sub model
    Comp_pool  [z ; Dev_pool]             Donor_Sub model   (pooled: c_dk := c_k)
    Dev_poolq  z - c_k^q                  Sub model
    Comp_poolq [z ; Dev_poolq]            Donor_Sub model   (pooled over the held-out donors' PBS cells, c_k^q)
    Adj_cf     z - (c_hat_dk - c_k)       Donor_Only model
    Dev_cf     z - c_hat_dk               Sub model
    Comp_cf    [Adj_cf ; Dev_cf]          Donor_Sub model

c_k: PBS cells of the training donors (as in training); c_k^q: PBS cells of the fold's held-out donors (the
query-derived pooled reference of unpaired datasets; Dev_poolq / Comp_poolq only); c_dk: the held-out donor's own PBS cells; c_hat_dk:
mean of the CellOT counterfactual controls of the donor's cytokine cells (map of the fold: held-out donors'
cytokine-treated cells -> training donors' PBS cells, cytokine identity and donor not used; cellot_maps.py
--target train), c_k where missing. --ck-from query / --cf-target heldout give the alternative designs.

Output: results/cellot/substituted_input_results.csv (other designs: ..._ck-{ck}_cf-{cf}.csv) (macro F1, macro one-vs-rest AUROC; overall
and per cell type).
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import cytap
from paper_utils import (FOLD_DONORS, cellot_map, compose_input, fold_references, load_classifier, load_parse,
                         macro_auroc, out, path)

ap = argparse.ArgumentParser()
ap.add_argument("--ck-from", choices=["train", "query"], default="train",
                help="PBS cells for c_k: training donors (train) or held-out donors (query)")
ap.add_argument("--cf-target", choices=["train", "heldout"], default="train",
                help="PBS cells the fold CellOT maps were trained onto: training or held-out donors")
args = ap.parse_args()
OUT_NAME = ("substituted_input_results.csv" if (args.ck_from, args.cf_target) == ("train", "train")
            else f"substituted_input_results_ck-{args.ck_from}_cf-{args.cf_target}.csv")

INPUTS = {  # name: (CV model, representation, reference used as c_k, reference used as c_dk)
    "Raw": ("Baseline", "raw", "ck", "cdk"),
    "Adj_true": ("Donor_Only", "adj", "ck", "cdk"),
    "Dev_true": ("Sub", "dev", "ck", "cdk"),
    "Comp_true": ("Donor_Sub", "comp", "ck", "cdk"),
    "Dev_pool": ("Sub", "dev", "ck", "ck"),
    "Comp_pool": ("Donor_Sub", "comp", "ck", "ck"),
    "Dev_poolq": ("Sub", "dev", "ckq", "ckq"),
    "Comp_poolq": ("Donor_Sub", "comp", "ckq", "ckq"),
    "Adj_cf": ("Donor_Only", "adj", "ck", "ccf"),
    "Dev_cf": ("Sub", "dev", "ck", "ccf"),
    "Comp_cf": ("Donor_Sub", "comp", "ck", "ccf"),
}


Z, obs = load_parse()
classes = np.array(cytap.load_model(str(path("model")), device="cpu").classes)
rows_out = []
for fold in FOLD_DONORS:
    ref = fold_references(Z, obs, fold, cellot_map(fold, target=args.cf_target), ck_from=args.ck_from)
    r = ref["rows"]
    z = Z[r]
    y = np.searchsorted(classes, obs["cytokine"].to_numpy()[r])
    ct = obs["cell_type"].to_numpy()[r]
    ck = ref["ck"][ref["ct_idx"]]
    refq = fold_references(Z, obs, fold, ck_from="query")  # pooled over the held-out donors' PBS cells
    assert np.array_equal(refq["rows"], r)
    tables = {"ck": ck, "ckq": refq["ck"][refq["ct_idx"]], "cdk": ref["cdk"][ref["grp_idx"]],
              "ccf": ref["ccf"][ref["grp_idx"]]}

    for name, (model_name, rep, kref, cref) in INPUTS.items():
        stem = path("cv_models") / f"{model_name}_Fold{fold}"
        if not (stem.with_suffix(".pt").exists() or stem.with_suffix(".pth").exists()):
            print(f"[skip] {name}: no model {stem}")
            continue
        model = load_classifier(stem)
        probs = cytap.predict_proba(model, compose_input(z, tables[kref], tables[cref], rep), batch_size=8192)
        pred = probs.argmax(1)
        rec = {"Input": name, "Model": model_name, "Fold": fold, "n_cells": len(y),
               "Accuracy": 100 * float((pred == y).mean()),
               "Macro_F1": f1_score(y, pred, average="macro", zero_division=0),
               "AUROC": macro_auroc(y, probs, len(classes))}
        for c in np.unique(ct):
            m = ct == c
            key = str(c).replace(" ", "_")
            rec[f"CellType_AUROC/{key}"] = macro_auroc(y[m], probs[m], len(classes))
            rec[f"CellType_Macro_F1/{key}"] = f1_score(y[m], pred[m], average="macro", zero_division=0)
        rows_out.append(rec)
        print(f"fold {fold} {name:10s} ({model_name}): AUROC {rec['AUROC']:.4f}  F1 {rec['Macro_F1']:.4f}")
        del model, probs
        torch.cuda.empty_cache()
    del z, tables, ck, refq
    gc.collect()

df = pd.DataFrame(rows_out)
df.to_csv(out("cellot", OUT_NAME), index=False)
summ = df.groupby("Input")[["AUROC", "Macro_F1"]].mean().reindex(list(INPUTS))
if {"Raw", "Comp_true"} <= set(summ.index):
    lo, hi = summ.loc["Raw", "AUROC"], summ.loc["Comp_true", "AUROC"]
    summ["AUROC_recovery_%"] = 100 * (summ["AUROC"] - lo) / (hi - lo)  # Raw = 0, Comp_true = 100
print(summ.round(4).to_string())
