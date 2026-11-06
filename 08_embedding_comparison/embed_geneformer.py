"""Geneformer V2-104M cell embeddings (768-d CLS token of the penultimate layer) of Parse-PBMC,
one chunk at a time (restartable).

Run with the Geneformer V2 environment's python:
    nohup python 08_embedding_comparison/embed_geneformer.py run > gf.log 2>&1 &
    python 08_embedding_comparison/embed_geneformer.py merge   # -> scfm.embeddings.Geneformer

Per chunk: gene symbols -> Ensembl IDs (Geneformer dictionaries; duplicate IDs summed), total
counts over all genes as n_counts, TranscriptomeTokenizer(model_version="V2"), then a direct
forward pass (BertModel, fp16 autocast, cells sorted by length and batched by token budget).
Same definition as EmbExtractor(emb_mode="cls", emb_layer=-1); the loop is written out so that
9.7M embeddings are saved as .npy with their cell IDs instead of CSV.
"""
import argparse
import glob
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_utils import chunks, merge, output, path

TOK_BUDGET = 64_000   # tokens per batch
MAX_BS = 256
NPROC = 16
WORK = path("scfm", "geneformer_work")
TMP_DIR, TOK_DIR, EMB_DIR = WORK / "tmp", WORK / "tokenized", WORK / "emb"


def build_gene_map(var, gf):
    """Matrix M (n_genes x n_vocab_genes) mapping genes to Geneformer vocabulary IDs (duplicates summed)."""
    load = lambda f: pickle.load(open(f, "rb"))  # noqa: E731
    tok, name2id, id_map = load(gf.TOKEN_DICTIONARY_FILE), load(gf.ENSEMBL_DICTIONARY_FILE), load(gf.ENSEMBL_MAPPING_FILE)
    ens_col = next((c for c in var.columns if var[c].astype(str).str.upper().str.startswith("ENSG").mean() > 0.5), None)
    if ens_col is not None:
        ens = var[ens_col].astype(str).str.upper().str.split(".").str[0]
    else:
        ens = pd.Series([name2id.get(g, name2id.get(g.upper())) for g in var.index])
    final = pd.Series([e if e in tok else None for e in (id_map.get(e) if isinstance(e, str) else None for e in ens)])
    keep = np.where(final.notna())[0]
    uniq, col = np.unique(final.iloc[keep].values.astype(str), return_inverse=True)
    M = sp.csr_matrix((np.ones(len(keep), np.float32), (keep, col)), shape=(len(var), len(uniq)))
    return M, uniq, tok["<pad>"]


def prep_chunk(f, name, M, uniq):
    a = ad.read_h5ad(f)
    X = sp.csr_matrix(a.X)
    ad.AnnData(
        X=(X @ M).tocsr().astype(np.float32),
        obs=pd.DataFrame({"n_counts": np.asarray(X.sum(1)).ravel(),   # total counts over all genes
                          "cell_id": a.obs_names.astype(str)}, index=a.obs_names),
        var=pd.DataFrame({"ensembl_id": uniq}, index=uniq),
    ).write_h5ad(TMP_DIR / f"{name}.h5ad")


def embed_chunk(name, model, pad, torch, load_from_disk):
    ds = load_from_disk(str(TOK_DIR / f"{name}.dataset")).sort("length", reverse=True)
    lengths = np.array(ds["length"])
    E = np.empty((len(ds), model.config.hidden_size), np.float32)
    i = 0
    with torch.no_grad():
        while i < len(ds):
            bs = int(max(1, min(MAX_BS, TOK_BUDGET // lengths[i])))
            b = ds[i:i + bs]["input_ids"]
            ids = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in b], batch_first=True,
                                                  padding_value=pad).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(input_ids=ids, attention_mask=(ids != pad).long()).hidden_states[-2]  # penultimate layer
            E[i:i + len(b)] = h[:, 0].float().cpu().numpy()                                    # CLS token
            i += len(b)
    for key, arr in (("cls", E), ("cell_id", np.array(ds["cell_id"]))):  # cell_id file = completion marker
        np.save(EMB_DIR / f"{name}_{key}.tmp.npy", arr)
        os.replace(EMB_DIR / f"{name}_{key}.tmp.npy", EMB_DIR / f"{name}_{key}.npy")


def run():
    import geneformer as gf
    import torch
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from transformers import BertModel

    for d in (TMP_DIR, TOK_DIR, EMB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    fs = chunks()
    M, uniq, pad = build_gene_map(ad.read_h5ad(fs[0], backed="r").var, gf)
    print(f"genes -> {len(uniq)} Geneformer vocabulary genes", flush=True)
    tk = TranscriptomeTokenizer({"cell_id": "cell_id"}, nproc=NPROC, model_version="V2")
    model = BertModel.from_pretrained(str(path("scfm", "geneformer_model")), output_hidden_states=True,
                                      add_pooling_layer=False, attn_implementation="sdpa").to("cuda").eval()
    for f in fs:
        name = f.stem
        if (EMB_DIR / f"{name}_cell_id.npy").exists():
            continue
        t0 = time.time()
        if not (TOK_DIR / f"{name}.done").exists():
            shutil.rmtree(TOK_DIR / f"{name}.dataset", ignore_errors=True)
            for old in glob.glob(str(TMP_DIR / "*.h5ad")):  # tokenizer reads every file in TMP_DIR
                os.remove(old)
            prep_chunk(f, name, M, uniq)
            tk.tokenize_data(str(TMP_DIR), str(TOK_DIR), name, file_format="h5ad")
            os.remove(TMP_DIR / f"{name}.h5ad")
            (TOK_DIR / f"{name}.done").touch()
        t1 = time.time()
        embed_chunk(name, model, pad, torch, load_from_disk)
        print(f"[{time.ctime()}] {name}: tokenize {(t1 - t0) / 60:.1f} min | embed {(time.time() - t1) / 60:.1f} min",
              flush=True)
    print("ALL DONE (tokenized/ can be deleted)")


def parts():
    for f in chunks():
        yield np.load(EMB_DIR / f"{f.stem}_cell_id.npy"), np.load(EMB_DIR / f"{f.stem}_cls.npy")


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["run", "merge"])
p.add_argument("--out", help="output .npy (default scfm.embeddings.Geneformer)")
a = p.parse_args()
if a.cmd == "run":
    run()
else:
    missing = [f.stem for f in chunks() if not (EMB_DIR / f"{f.stem}_cell_id.npy").exists()]
    if missing:
        sys.exit(f"unfinished chunks: {missing}")
    merge(parts(), output("Geneformer", a.out))
