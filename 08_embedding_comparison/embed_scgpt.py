"""scGPT whole-human cell embeddings (512-d) with scgpt.tasks.embed_data.

Run with the scGPT environment's python:
    python 08_embedding_comparison/embed_scgpt.py [--out FILE]      # -> scfm.embeddings.scGPT

Input: log-normalized expression (scGPT bins the values of each cell). Parse-PBMC is embedded in
chunks of 50,000 cells (embed_data batch size 64); the output .h5ad holds obs and
obsm['X_scGPT'] (X dropped). embed_data samples genes at random for cells with more expressed
genes than its max_length, so the torch and numpy seeds are fixed here; the paper embeddings
were made without fixed seeds.
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import h5py
import numpy as np
import scgpt as scg
import torch

from scfm_utils import output, path

CHUNK, BATCH, SEED = 50_000, 64, 0


def embed(adata, gene_col):
    out = scg.tasks.embed_data(adata_or_file=adata, model_dir=path("scfm", "scgpt_model"), gene_col=gene_col,
                               batch_size=BATCH, device="cuda", return_new_adata=False)
    return np.asarray(out.obsm["X_scGPT"], dtype=np.float32)


p = argparse.ArgumentParser()
p.add_argument("--out", help="output .h5ad (default scfm.embeddings.scGPT)")
a = p.parse_args()
torch.manual_seed(SEED)
np.random.seed(SEED)
src, dst = path("parse", "normalized"), output("scGPT", a.out)

with h5py.File(src, "r") as h:
    if "uns/log1p/base" in h:
        sys.exit(f"{src}: uns['log1p']['base'] cannot be read by the anndata of the scGPT environment; "
                 "remove it from a copy of the file (h5py: del f['uns/log1p/base'])")

A = ad.read_h5ad(src, backed="r")
gene_col = "gene_symbol"
parts = []
for s in range(0, A.n_obs, CHUNK):
    chunk = A[s:min(s + CHUNK, A.n_obs)].to_memory()
    if gene_col not in chunk.var:
        chunk.var[gene_col] = chunk.var_names
    parts.append(embed(chunk, gene_col))
    print(f"cells {s}-{s + chunk.n_obs}: {parts[-1].shape}", flush=True)
    del chunk
    gc.collect()

out = ad.AnnData(obs=A.obs.copy())
out.obsm["X_scGPT"] = np.vstack(parts)
out.write_h5ad(dst)
print(f"saved {dst}: {out.obsm['X_scGPT'].shape}")
