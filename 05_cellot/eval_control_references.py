"""Control references for unpaired inference, evaluated on Parse-PBMC held-out donors.

The 4-fold CV models (trained with true donor-matched centroids, 06_input_ablation) are applied
without retraining to the held-out donors' cytokine cells, with the control centroid replaced:

    Raw        z                          Baseline model
    Adj_true   z - (c_dk - c_k)           Donor_Only model
    Dev_true   z - c_dk                   Sub model
    Comp_true  [Adj_true ; Dev_true]      Donor_Sub model
    Dev_pool   z - c_k                    Sub model
    Comp_pool  [z ; Dev_pool]             Donor_Sub model   (pooled: c_dk := c_k)
    Adj_cf     z - (c_hat_dk - c_k)       Donor_Only model
    Dev_cf     z - c_hat_dk               Sub model
    Comp_cf    [Adj_cf ; Dev_cf]          Donor_Sub model

c_k: training donors' PBS cells; c_dk: the held-out donor's own PBS cells; c_hat_dk: mean of the CellOT
counterfactual controls of the donor's cytokine cells (map of the fold: cytokine-treated -> PBS cells of the
held-out donors, cytokine identity not used; cellot_maps.py), c_k where missing.

Output: results/cellot/substituted_input_results.csv (macro F1, macro one-vs-rest AUROC; overall
and per cell type).
"""
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

INPUTS = {  # name: (CV model, representation, reference used as c_dk)
    "Raw": ("Baseline", "raw", "cdk"),
    "Adj_true": ("Donor_Only", "adj", "cdk"),
    "Dev_true": ("Sub", "dev", "cdk"),
    "Comp_true": ("Donor_Sub", "comp", "cdk"),
    "Dev_pool": ("Sub", "dev", "ck"),
    "Comp_pool": ("Donor_Sub", "comp", "ck"),
    "Adj_cf": ("Donor_Only", "adj", "ccf"),
    "Dev_cf": ("Sub", "dev", "ccf"),
    "Comp_cf": ("Donor_Sub", "comp", "ccf"),
}


Z, obs = load_parse()
classes = np.array(cytap.load_model(str(path("model")), device="cpu").classes)
rows_out = []
for fold in FOLD_DONORS:
    ref = fold_references(Z, obs, fold, cellot_map(fold))
    r = ref["rows"]
    z = Z[r]
    y = np.searchsorted(classes, obs["cytokine"].to_numpy()[r])
    ct = obs["cell_type"].to_numpy()[r]
    ck = ref["ck"][ref["ct_idx"]]
    tables = {"ck": ck, "cdk": ref["cdk"][ref["grp_idx"]], "ccf": ref["ccf"][ref["grp_idx"]]}

    for name, (model_name, rep, cref) in INPUTS.items():
        stem = path("cv_models") / f"{model_name}_Fold{fold}"
        if not (stem.with_suffix(".pt").exists() or stem.with_suffix(".pth").exists()):
            print(f"[skip] {name}: no model {stem}")
            continue
        model = load_classifier(stem)
        probs = cytap.predict_proba(model, compose_input(z, ck, tables[cref], rep), batch_size=8192)
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
    del z, tables, ck
    gc.collect()

df = pd.DataFrame(rows_out)
df.to_csv(out("cellot", "substituted_input_results.csv"), index=False)
summ = df.groupby("Input")[["AUROC", "Macro_F1"]].mean().reindex(list(INPUTS))
if {"Raw", "Comp_true"} <= set(summ.index):
    lo, hi = summ.loc["Raw", "AUROC"], summ.loc["Comp_true", "AUROC"]
    summ["AUROC_recovery_%"] = 100 * (summ["AUROC"] - lo) / (hi - lo)  # Raw = 0, Comp_true = 100
print(summ.round(4).to_string())
