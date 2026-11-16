"""Control references for unpaired inference, evaluated on Parse-PBMC held-out donors.

The 4-fold CV models (trained with true donor-matched centroids, 06_input_ablation) are applied
without retraining to the held-out donors' cytokine cells, with the control centroid replaced:

    Input              representation          model        Reference  Source of the control cells
    Raw                z                       Baseline     none       -
    Adj/Dev/Comp_true  c_dk                    *            true       held-out donor's own PBS cells
    Dev/Comp_pool_train  c_dk := c_k           Sub/Donor_Sub pooled    training donors' PBS cells (as in training)
    Dev/Comp_pool_heldout c_dk := c_k := c_k^q Sub/Donor_Sub pooled    held-out donors' PBS cells (query-derived,
                                                                       as for unpaired datasets)
    Adj/Dev/Comp_cf_train   c_dk := c_hat_dk   *            CellOT     map onto the training donors' PBS cells
    Adj/Dev/Comp_cf_heldout c_dk := c_hat_dk   *            CellOT     map onto the held-out donors' PBS cells

    Adj = z - (c_dk - c_k) (Donor_Only model), Dev = z - c_dk (Sub model), Comp = [Adj ; Dev] (Donor_Sub model).

c_k: PBS cells of the training donors unless stated; c_hat_dk: mean of the CellOT counterfactual controls of the
held-out donor's cytokine cells of cell type k (c_k where missing). Both CellOT maps of a fold transport the
held-out donors' cytokine-treated cells (cytokines pooled, no donor labels); they differ only in the PBS cells they
were trained onto (cellot_maps.py --target train / heldout).

Output: results/cellot/control_reference_evaluation.csv, one row per fold and input, with columns Reference and
Control_source; macro F1 and macro one-vs-rest AUROC, overall and per cell type.

python 05_cellot/eval_control_references.py
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

OUT_NAME = "control_reference_evaluation.csv"
TRAIN_SRC, HELD_SRC = "training donors' PBS", "held-out donors' PBS"
INPUTS = {  # name: (CV model, representation, c_k, c_dk, Reference, Control_source)
    "Raw": ("Baseline", "raw", "ck", "cdk", "none", "-"),
    "Adj_true": ("Donor_Only", "adj", "ck", "cdk", "true", "held-out donor's own PBS"),
    "Dev_true": ("Sub", "dev", "ck", "cdk", "true", "held-out donor's own PBS"),
    "Comp_true": ("Donor_Sub", "comp", "ck", "cdk", "true", "held-out donor's own PBS"),
    "Dev_pool_train": ("Sub", "dev", "ck", "ck", "pooled", TRAIN_SRC),
    "Comp_pool_train": ("Donor_Sub", "comp", "ck", "ck", "pooled", TRAIN_SRC),
    "Dev_pool_heldout": ("Sub", "dev", "ckq", "ckq", "pooled", HELD_SRC),
    "Comp_pool_heldout": ("Donor_Sub", "comp", "ckq", "ckq", "pooled", HELD_SRC),
    "Adj_cf_train": ("Donor_Only", "adj", "ck", "cf_train", "CellOT", TRAIN_SRC),
    "Dev_cf_train": ("Sub", "dev", "ck", "cf_train", "CellOT", TRAIN_SRC),
    "Comp_cf_train": ("Donor_Sub", "comp", "ck", "cf_train", "CellOT", TRAIN_SRC),
    "Adj_cf_heldout": ("Donor_Only", "adj", "ck", "cf_heldout", "CellOT", HELD_SRC),
    "Dev_cf_heldout": ("Sub", "dev", "ck", "cf_heldout", "CellOT", HELD_SRC),
    "Comp_cf_heldout": ("Donor_Sub", "comp", "ck", "cf_heldout", "CellOT", HELD_SRC),
}


Z, obs = load_parse()
classes = np.array(cytap.load_model(str(path("model")), device="cpu").classes)
rows_out = []
for fold in FOLD_DONORS:
    ref = fold_references(Z, obs, fold, cellot_map(fold, target="train"))
    refh = fold_references(Z, obs, fold, cellot_map(fold, target="heldout"))
    refq = fold_references(Z, obs, fold, ck_from="query")  # pooled over the held-out donors' PBS cells
    r = ref["rows"]
    assert np.array_equal(refh["rows"], r) and np.array_equal(refq["rows"], r)
    z = Z[r]
    y = np.searchsorted(classes, obs["cytokine"].to_numpy()[r])
    ct = obs["cell_type"].to_numpy()[r]
    g = ref["grp_idx"]
    tables = {"ck": ref["ck"][ref["ct_idx"]], "ckq": refq["ck"][refq["ct_idx"]], "cdk": ref["cdk"][g],
              "cf_train": ref["ccf"][g], "cf_heldout": refh["ccf"][refh["grp_idx"]]}

    for name, (model_name, rep, kref, cref, reference, source) in INPUTS.items():
        stem = path("cv_models") / f"{model_name}_Fold{fold}"
        if not (stem.with_suffix(".pt").exists() or stem.with_suffix(".pth").exists()):
            print(f"[skip] {name}: no model {stem}")
            continue
        model = load_classifier(stem)
        probs = cytap.predict_proba(model, compose_input(z, tables[kref], tables[cref], rep), batch_size=8192)
        pred = probs.argmax(1)
        rec = {"Input": name, "Reference": reference, "Control_source": source, "Model": model_name,
               "Fold": fold, "n_cells": len(y),
               "Accuracy": 100 * float((pred == y).mean()),
               "Macro_F1": f1_score(y, pred, average="macro", zero_division=0),
               "AUROC": macro_auroc(y, probs, len(classes))}
        for c in np.unique(ct):
            m = ct == c
            key = str(c).replace(" ", "_")
            rec[f"CellType_AUROC/{key}"] = macro_auroc(y[m], probs[m], len(classes))
            rec[f"CellType_Macro_F1/{key}"] = f1_score(y[m], pred[m], average="macro", zero_division=0)
        rows_out.append(rec)
        print(f"fold {fold} {name:18s} ({model_name}): AUROC {rec['AUROC']:.4f}  F1 {rec['Macro_F1']:.4f}")
        del model, probs
        torch.cuda.empty_cache()
    del z, tables, ref, refh, refq
    gc.collect()

df = pd.DataFrame(rows_out)
df.to_csv(out("cellot", OUT_NAME), index=False)
summ = df.groupby("Input")[["AUROC", "Macro_F1"]].mean().reindex(list(INPUTS))
if {"Raw", "Comp_true"} <= set(summ.index):
    lo, hi = summ.loc["Raw", "AUROC"], summ.loc["Comp_true", "AUROC"]
    summ["AUROC_recovery_%"] = 100 * (summ["AUROC"] - lo) / (hi - lo)  # Raw = 0, Comp_true = 100
print(summ.round(4).to_string())
