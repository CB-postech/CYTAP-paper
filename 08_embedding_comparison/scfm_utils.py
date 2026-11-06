"""Helpers shared by the embedding scripts (run in each model's own environment).

python 08_embedding_comparison/scfm_utils.py chunks   # parse.counts -> 250,000-cell chunks (UCE, Geneformer, scFoundation)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_utils import path  # noqa: E402


def read_obs_names(f) -> pd.Index:
    """obs index of an .h5ad, read with h5py (independent of the anndata version)."""
    with h5py.File(f, "r") as h:
        key = h["obs"].attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else key
        idx = h["obs"][key]
        if isinstance(idx, h5py.Group):  # nullable-string-array (newer anndata)
            idx = idx["values"]
        return pd.Index(idx.asstr()[...])


def chunks():
    fs = sorted(path("scfm", "chunks").glob("chunk_*.h5ad"))
    if not fs:
        sys.exit(f"no chunks in {path('scfm', 'chunks')}: run 08_embedding_comparison/scfm_utils.py chunks")
    return fs


def output(model, out=None) -> Path:
    """Output .npy; an existing file is never overwritten."""
    f = Path(out) if out else path("scfm", "embeddings", model)
    if f.exists():
        sys.exit(f"{f} exists; pass --out to write a new file")
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def merge(parts, out_file):
    """Write (cell_ids, embedding) parts to one (n_src x d) float32 .npy in the row order of
    parse.counts. ``parts`` yields (ids, array) pairs."""
    orig = read_obs_names(path("parse", "counts"))
    E, filled = None, np.zeros(len(orig), bool)
    for ids, emb in parts:
        if E is None:
            E = np.lib.format.open_memmap(out_file, mode="w+", dtype=np.float32, shape=(len(orig), emb.shape[1]))
        idx = orig.get_indexer(pd.Index(np.asarray(ids).astype(str)))
        assert (idx >= 0).all() and len(idx) == len(emb), "cell ids do not match parse.counts"
        E[idx] = emb
        filled[idx] = True
    E.flush()
    s = np.sort(np.random.default_rng(0).choice(len(orig), min(10_000, len(orig)), replace=False))
    sample = np.asarray(E[s])
    print(f"saved {out_file} {E.shape} | missing cells {(~filled).sum()} | "
          f"NaN (10k sample) {int(np.isnan(sample).sum())} | zero rows {(np.abs(sample).sum(1) == 0).sum()}")


def make_chunks(size=250_000):
    """Split the raw-count Parse-PBMC file into chunks; existing chunks are kept."""
    import os

    import anndata as ad
    d = path("scfm", "chunks")
    d.mkdir(parents=True, exist_ok=True)
    A = ad.read_h5ad(path("parse", "counts"), backed="r")
    for i, s in enumerate(range(0, A.n_obs, size)):
        f = d / f"chunk_{i:03d}.h5ad"
        if f.exists():
            continue
        sub = A[s:min(s + size, A.n_obs)].to_memory()
        ad.AnnData(X=sub.X, obs=sub.obs, var=sub.var).write_h5ad(f"{f}.tmp")
        os.replace(f"{f}.tmp", f)  # no partial files if interrupted
        print(f"{f.name}: {sub.shape}", flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["chunks"]:
    make_chunks()
