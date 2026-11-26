"""Donor-held-out 4-fold CV with other single-cell foundation model embeddings.

Same folds, labels, classifier, hyperparameters and metrics as the State input ablation
(06_input_ablation/input_ablation.py cv); only the cell embedding z is replaced.

  raw  (Baseline)    z                         all embeddings (UCE, Geneformer, scFoundation, scGPT)
  adj  (Donor_Only)  z - c_{d,k} + c_k         scFoundation
  dev  (Sub)         z - c_{d,k}
  comp (Donor_Sub)   [adj ; dev]

c_k: PBS centroid of cell type k over the training donors; c_{d,k}: each donor's own PBS centroid
(c_k if none). Cells whose embedding is not finite are excluded.

python 08_embedding_comparison/cv_embeddings.py --embeddings UCE Geneformer scFoundation scGPT --inputs raw
python 08_embedding_comparison/cv_embeddings.py --embeddings scFoundation --inputs adj dev comp
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

from cytap.model import CytokineClassifier, FocalLoss
from paper_utils import FOLD_DONORS, TABLE_NAMES, macro_auroc, out, path
from scfm_utils import read_obs_names

DONOR, CT, COND, PBS = "donor", "cell_type", "cytokine", "PBS"
NAMES = {k: v for k, v in TABLE_NAMES.items() if k != "hvg"}
PARAMS = {"batch_size": 4096, "dropout": 0.3, "focal_gamma": 2, "learning_rate": 1.4351e-4, "epochs": 10}
HIDDEN = (2048, 1024)
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

p = argparse.ArgumentParser()
p.add_argument("--embeddings", nargs="+", required=True, help="keys of scfm.embeddings in config/paths.yaml")
p.add_argument("--inputs", nargs="+", choices=list(NAMES), default=["raw"])
args = p.parse_args()

OUT_CSV = out("scfm_embeddings", "embedding_comparison_results.csv")
MODEL_DIR = out("scfm_embeddings", "models", "x").parent
rows = pd.read_csv(OUT_CSV).to_dict("records") if OUT_CSV.exists() else []


# ---------------------------------------------------------------- embeddings
def load_npy(f, ref, src, block=500_000):
    """(n_src x d) .npy in parse.counts row order -> rows in ref order."""
    E = np.load(f, mmap_mode="r")
    assert E.shape[0] == len(src), f"{f}: {E.shape[0]} rows vs {len(src)} cells in parse.counts"
    idx = src.get_indexer(ref)
    assert (idx >= 0).all(), f"{(idx < 0).sum()} cells not found in parse.counts"
    X = np.empty((len(idx), E.shape[1]), dtype=np.float32)
    for i in range(0, len(idx), block):
        b = idx[i:i + block]
        o = np.argsort(b)  # read in file order
        X[i:i + block][o] = E[b[o]]
    return X


def load_obsm(f, key, ref, block=200_000):
    """obsm[key] of an .h5ad, aligned to ref by obs_names; cells missing from the file are NaN."""
    names = read_obs_names(f)
    target = pd.Series(np.arange(len(ref)), index=ref).reindex(names).to_numpy()
    with h5py.File(f, "r") as h:
        ds = h["obsm"][key]
        X = np.full((len(ref), ds.shape[1]), np.nan, dtype=np.float32)
        for i in range(0, ds.shape[0], block):
            t = target[i:i + block]
            m = ~np.isnan(t)
            if m.any():
                X[t[m].astype(np.int64)] = ds[i:i + block][m]
    n_missing = int(np.isnan(X[:, 0]).sum())
    if n_missing > 0.01 * len(ref):
        raise ValueError(f"{n_missing} of {len(ref)} cells missing from {f}: check obs_names")
    return X


def load(name, ref, src):
    f = path("scfm", "embeddings", name)
    return load_obsm(f, "X_scGPT", ref) if f.suffix == ".h5ad" else load_npy(f, ref, src)


def finite_rows(X, block=500_000):
    ok = np.zeros(len(X), dtype=bool)
    for i in range(0, len(X), block):
        ok[i:i + block] = np.isfinite(X[i:i + block]).all(1)
    return ok


# ---------------------------------------------------------------- loaders
G = {}  # tensors shared with forked loader workers


class GatherRows(Dataset):
    """Batch of positions -> (z, donor x cell-type group, cell type, label)."""

    def __init__(self, cell_idx):
        self.cell_idx = torch.from_numpy(np.asarray(cell_idx, dtype=np.int64))

    def __len__(self):
        return len(self.cell_idx)

    def __getitem__(self, positions):
        c = self.cell_idx[torch.as_tensor(positions, dtype=torch.long)]
        return G["E"][c], G["grp"][c], G["ct"][c], G["y"][c]


def gather_loader(cell_idx, train):
    ds = GatherRows(cell_idx)
    return DataLoader(ds, sampler=BatchSampler(RandomSampler(ds) if train else SequentialSampler(ds),
                                               PARAMS["batch_size"], drop_last=train),
                      batch_size=None, num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=NUM_WORKERS > 0, multiprocessing_context="fork")


def tensor_loader(X, y, train):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=PARAMS["batch_size"],
                      shuffle=train, num_workers=NUM_WORKERS, pin_memory=True, drop_last=train)


def featurizer(rep, cdk=None, ck=None):
    """Batch on the GPU -> model input. adj = (z - c_{d,k}) + c_k, as in the original run."""
    def f(batch):
        b = [t.to(DEVICE, non_blocking=True) for t in batch]
        if rep == "raw" and len(b) == 2:
            return b
        x, g, c, y = b
        if rep == "raw":
            return x, y
        dev = x - cdk[g]
        if rep == "dev":
            return dev, y
        adj = dev + ck[c]
        return (adj if rep == "adj" else torch.cat([adj, dev], dim=1)), y
    return f


# ---------------------------------------------------------------- training
def train_eval(run, train_loader, test_loader, featurize, in_dim, n_classes, ct_test):
    """Train for PARAMS['epochs'] epochs; test metrics after every epoch."""
    torch.manual_seed(42)
    model = CytokineClassifier(in_dim, n_classes, HIDDEN, PARAMS["dropout"]).to(DEVICE)
    net = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.AdamW(model.parameters(), lr=PARAMS["learning_rate"])
    crit = FocalLoss(gamma=PARAMS["focal_gamma"]).to(DEVICE)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=PARAMS["learning_rate"],
                                                total_steps=len(train_loader) * PARAMS["epochs"], pct_start=0.3)
    history = []
    for epoch in range(1, PARAMS["epochs"] + 1):
        net.train()
        tr_loss = n = top1 = top5 = 0
        for batch in train_loader:
            xb, yb = featurize(batch)
            opt.zero_grad()
            logits = net(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item()
            t5 = logits.topk(5, dim=1).indices
            top1 += (t5[:, 0] == yb).sum().item()
            top5 += (t5 == yb[:, None]).any(1).sum().item()
            n += len(yb)

        net.eval()
        te_loss, probs, ys = 0.0, [], []
        with torch.no_grad():
            for batch in test_loader:
                xb, yb = featurize(batch)
                logits = net(xb)
                te_loss += crit(logits, yb).item()
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
                ys.append(yb.cpu().numpy())
        probs, y_te = np.concatenate(probs), np.concatenate(ys)
        pred = probs.argmax(1)
        top5_te = np.argsort(-probs, axis=1)[:, :5]
        m = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
             "train_loss": tr_loss / len(train_loader), "test_loss": te_loss / len(test_loader),
             "train_accuracy": 100 * top1 / n, "test_accuracy": 100 * float((pred == y_te).mean()),
             "train_accuracy_top5": 100 * top5 / n,
             "test_accuracy_top5": 100 * float((top5_te == y_te[:, None]).any(1).mean()),
             "Overall/Macro_F1": f1_score(y_te, pred, average="macro", zero_division=0),
             "Overall/AUROC": macro_auroc(y_te, probs, n_classes)}
        for ct in np.unique(ct_test):
            s = ct_test == ct
            key = str(ct).replace(" ", "_")
            m[f"CellType_Macro_F1/{key}"] = f1_score(y_te[s], pred[s], average="macro", zero_division=0)
            m[f"CellType_AUROC/{key}"] = macro_auroc(y_te[s], probs[s], n_classes)
        print(f"{run} epoch {epoch:02d} | train loss {m['train_loss']:.4f} | "
              f"test macro F1 {m['Overall/Macro_F1']:.4f} AUROC {m['Overall/AUROC']:.4f}", flush=True)
        history.append(m)
    return model, history


# ---------------------------------------------------------------- main
a = ad.read_h5ad(path("parse", "emb"), backed="r")
ref = pd.Index(a.obs_names.astype(str))
obs = a.obs[[DONOR, CT, COND]].astype(str).reset_index(drop=True)
del a
src = read_obs_names(path("parse", "counts"))
pbs = (obs[COND] == PBS).to_numpy()
classes, y_cyt = np.unique(obs[COND].to_numpy()[~pbs], return_inverse=True)
y_all = np.zeros(len(obs), dtype=np.int64)
y_all[~pbs] = y_cyt
ct_code, ct_names = pd.factorize(obs[CT].to_numpy())
d_code, d_names = pd.factorize(obs[DONOR].to_numpy())
n_ct = len(ct_names)
grp_code = d_code.astype(np.int64) * n_ct + ct_code
ct_all = obs[CT].to_numpy()

for emb in args.embeddings:
    todo = [(r, f) for r in args.inputs for f in FOLD_DONORS
            if not (MODEL_DIR / f"{emb}_{NAMES[r]}_Fold{f}.pth").exists()]
    if not todo:
        print(f"{emb}: all runs done")
        continue
    Z = load(emb, ref, src)
    valid = finite_rows(Z)
    print(f"{emb}: {Z.shape} | non-finite rows {(~valid).sum()} (excluded)", flush=True)
    G.update(E=torch.from_numpy(Z), grp=torch.from_numpy(grp_code), ct=torch.from_numpy(ct_code.astype(np.int64)),
             y=torch.from_numpy(y_all))

    for rep, fold in todo:
        run = f"{emb}_{NAMES[rep]}_Fold{fold}"
        test = obs[DONOR].isin(FOLD_DONORS[fold]).to_numpy()
        tr_idx = np.flatnonzero(~pbs & ~test & valid)
        te_idx = np.flatnonzero(~pbs & test & valid)
        print(f"\n=== {run}: train {len(tr_idx)} | test {len(te_idx)} cells", flush=True)
        rows = [r for r in rows if not (r["Embedding"] == emb and r["Condition"] == NAMES[rep] and int(r["Fold"]) == fold)]

        if rep == "raw":
            X_tr, X_te = np.ascontiguousarray(Z[tr_idx]), np.ascontiguousarray(Z[te_idx])
            loaders = tensor_loader(X_tr, y_all[tr_idx], True), tensor_loader(X_te, y_all[te_idx], False)
            feat, in_dim = featurizer("raw"), Z.shape[1]
        else:
            pbs_idx = np.flatnonzero(pbs & valid)
            ck = np.zeros((n_ct, Z.shape[1]), dtype=np.float32)
            for k in range(n_ct):
                i = pbs_idx[(ct_code[pbs_idx] == k) & ~test[pbs_idx]]
                if len(i):
                    ck[k] = Z[i].mean(axis=0)
            cdk = np.empty((len(d_names) * n_ct, Z.shape[1]), dtype=np.float32)
            for g in range(len(cdk)):
                i = pbs_idx[grp_code[pbs_idx] == g]
                cdk[g] = Z[i].mean(axis=0) if len(i) else ck[g % n_ct]
            loaders = gather_loader(tr_idx, True), gather_loader(te_idx, False)
            feat = featurizer(rep, torch.from_numpy(cdk).to(DEVICE), torch.from_numpy(ck).to(DEVICE))
            in_dim = Z.shape[1] * (2 if rep == "comp" else 1)

        model, history = train_eval(run, *loaders, feat, in_dim, len(classes), ct_all[te_idx])
        torch.save(model.state_dict(), MODEL_DIR / f"{run}.pth")
        rows.extend({"Embedding": emb, "Condition": NAMES[rep], "Fold": fold, **h} for h in history)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        del loaders, model
        if rep == "raw":
            del X_tr, X_te
        torch.cuda.empty_cache()
        gc.collect()
    G.clear()
    del Z
    gc.collect()
print(f"done -> {OUT_CSV}")
