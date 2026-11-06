"""UCE cell embeddings (4-layer model, 1,280-d) of Parse-PBMC with the official
eval_single_anndata.py, one 250,000-cell chunk at a time (restartable).

Run with the UCE environment's python:
    nohup python 08_embedding_comparison/embed_uce.py run --gpu 0 > uce.log 2>&1 &
    python 08_embedding_comparison/embed_uce.py merge          # -> scfm.embeddings.UCE

Settings: --species human --batch_size 50 --filter "" (cell/gene filtering off, so every cell
is embedded); model and number of layers are the repository defaults (4 layers).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad

from scfm_utils import chunks, merge, output, path

BATCH_SIZE = 50
UCE = path("scfm", "uce_dir")
OUT_DIR, LOG_DIR = UCE / "uce_output", UCE / "uce_logs"


def result(name):
    return OUT_DIR / f"{name}_uce_adata.h5ad"


def run(gpu):
    OUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for f in chunks():
        name = f.stem
        if result(name).exists():
            continue
        print(f"[{time.ctime()}] start {name}", flush=True)
        with open(LOG_DIR / f"{name}.log", "w") as log:
            subprocess.run([sys.executable, "eval_single_anndata.py", "--adata_path", str(f), "--dir", f"{OUT_DIR}/",
                            "--species", "human", "--batch_size", str(BATCH_SIZE), "--filter", ""],
                           cwd=UCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if not result(name).exists():
            sys.exit(f"FAILED {name}: see {LOG_DIR / f'{name}.log'}")
        for tmp in (OUT_DIR / f"{name}_counts.npz", OUT_DIR / f"{name}_proc.h5ad"):
            tmp.unlink(missing_ok=True)
        print(f"[{time.ctime()}] done {name}", flush=True)
    print("ALL DONE")


def parts():
    for f in chunks():
        a = ad.read_h5ad(result(f.stem))
        yield a.obs_names, a.obsm["X_uce"]


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["run", "merge"])
p.add_argument("--gpu", default="0")
p.add_argument("--out", help="output .npy (default scfm.embeddings.UCE)")
a = p.parse_args()
if a.cmd == "run":
    run(a.gpu)
else:
    missing = [f.stem for f in chunks() if not result(f.stem).exists()]
    if missing:
        sys.exit(f"unfinished chunks: {missing}")
    merge(parts(), output("UCE", a.out))
