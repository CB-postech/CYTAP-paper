"""CYTAP on the Parse-PBMC benchmark: out-of-fold (OOF) predictions of the 4-fold donor-held-out
Comp models, scored like the other methods.

Each held-out donor's cells (cytokine and PBS) are predicted by the model of its fold, with c_k from
the training donors' PBS cells and c_{d,k} from the donor's own PBS cells (c_k if none), as in training.
Per responsive condition (cell type C, cytokine X): donor-level median logit per class in X and in
PBS (donors with both), delta = mean over donors of X - PBS, all 90 classes ranked by delta;
true_rank = rank of X.

Inputs: Comp CV models {cv_models}/Donor_Sub_Fold{k}.pth (or .pt); responsive_conditions.csv
Output: CYTAP_output/cytap_top10_results.csv, cytap_classes.txt (ranking pool)

python 09_benchmark_parse/cytap_oof.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from cytap.predict import predict_proba
from paper_utils import FOLD_DONORS, bench, load_classifier, load_parse, path

PBS, TOP_N, EPS = "PBS", 10, 1e-6

Z, obs = load_parse()
donor, ct, cyt = (obs[c].to_numpy() for c in ("donor", "cell_type", "cytokine"))
pbs = cyt == PBS
classes = np.unique(cyt[~pbs])  # sorted, = model output order (LabelEncoder)
cts, donors = np.unique(ct), np.unique(donor)

oof = np.full((len(obs), len(classes)), np.nan, dtype=np.float32)
for fold, test_donors in FOLD_DONORS.items():
    model = load_classifier(path("cv_models") / f"Donor_Sub_Fold{fold}")
    test = np.isin(donor, test_donors)
    ck = {c: (Z[np.flatnonzero((ct == c) & pbs & ~test)].mean(0) if ((ct == c) & pbs & ~test).any()
              else np.zeros(Z.shape[1], np.float32)) for c in cts}
    cdk = {}
    for d in donors:
        for c in cts:
            m = (donor == d) & (ct == c) & pbs
            cdk[(d, c)] = Z[np.flatnonzero(m)].mean(0) if m.any() else ck[c]
    idx = np.flatnonzero(test)  # cytokine and PBS cells (PBS is the reference of the delta)
    for s in range(0, len(idx), 500_000):
        r = idx[s:s + 500_000]
        dev = Z[r] - np.stack([cdk[k] for k in zip(donor[r], ct[r])])
        X = np.empty((len(r), 2 * Z.shape[1]), dtype=np.float32)
        X[:, Z.shape[1]:] = dev                                       # Dev = z - c_{d,k}
        X[:, :Z.shape[1]] = dev + np.stack([ck[c] for c in ct[r]])    # Adj = (z - c_{d,k}) + c_k
        oof[r] = predict_proba(model, X, batch_size=8192)
    print(f"fold {fold}: {len(idx)} cells", flush=True)
    del model
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- scoring
p = np.clip(oof, EPS, 1 - EPS)
L = pd.DataFrame(np.log(p / (1 - p)), columns=classes)
L["donor"], L["cell_type"], L["cytokine"] = donor, ct, cyt
med = (L.dropna(subset=list(classes)).groupby(["cell_type", "cytokine", "donor"], observed=True)[list(classes)]
       .median())

resp = pd.read_csv(path("benchmark", "parse_dir") / "benchmark" / "responsive_conditions.csv")
rows = []
for c, x in resp[["celltype", "cytokine"]].itertuples(index=False):
    if x not in classes:
        rows.append((c, x, np.nan, "not_in_classes"))
        continue
    try:
        stim, ctl = med.loc[(c, x)], med.loc[(c, PBS)]
    except KeyError:
        rows.append((c, x, np.nan, "no_cells"))
        continue
    common = stim.index.intersection(ctl.index)
    if len(common) == 0:
        rows.append((c, x, np.nan, "no_cells"))
        continue
    delta = stim.loc[common].mean() - ctl.loc[common].mean()
    rows.append((c, x, list(delta.sort_values(ascending=False).index).index(x) + 1, "scored"))

res = pd.DataFrame(rows, columns=["celltype", "cytokine", "true_rank", "status"])
res["hit"] = res["true_rank"] <= TOP_N
res.to_csv(bench("parse_dir", "CYTAP_output", "cytap_top10_results.csv", new=True), index=False)
pd.Series(classes).to_csv(bench("parse_dir", "CYTAP_output", "cytap_classes.txt"), index=False, header=False)
print(res["status"].value_counts().to_string())
print(f"top-5 recall {(res.true_rank <= 5).mean():.3f} | top-10 recall {(res.true_rank <= 10).mean():.3f}")
