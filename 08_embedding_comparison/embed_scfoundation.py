"""scFoundation cell embeddings (3,072-d) of Parse-PBMC with the official get_embedding.py, per 250,000-cell chunk
(restartable: finished chunks are skipped).

Official settings: --input_type singlecell --output_type cell --pool_type all --tgthighres t4 --pre_normalized F
--version ce. Each chunk is first reordered to the 19,264 scFoundation genes (missing genes = 0), because
get_embedding.py skips its gene selection when a file has more genes than that.

Run with the python of an environment where scFoundation runs; several processes (GPUs) can share the chunks:
  python 08_embedding_comparison/embed_scfoundation.py run --gpu 0 --part 0 --nparts 2
  python 08_embedding_comparison/embed_scfoundation.py run --gpu 1 --part 1 --nparts 2
  python 08_embedding_comparison/embed_scfoundation.py merge          # -> scfm.embeddings.scFoundation
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_utils import chunks, merge, output, path, read_obs_names

SCF = path("scfm", "scfoundation_dir")
WORK = path("scfm", "scfoundation_work")
OUT_DIR, PREP_DIR, LOG_DIR = WORK / "official", WORK / "prep", WORK / "logs"
SUFFIX = "_01B-resolution_singlecell_cell_embedding_t4_resolution.npy"


def result(f):
    return OUT_DIR / f"{f.stem}{SUFFIX}"


def patch_load_py():
    """torch >= 2.6 defaults to weights_only=True, which cannot load the checkpoint; patch load.py (idempotent)."""
    p = SCF / "load.py"
    s = p.read_text()
    old = "model_data = torch.load(best_ckpt_path,map_location='cpu')"
    if old in s:
        p.write_text(s.replace(old, "model_data = torch.load(best_ckpt_path,map_location='cpu',weights_only=False)"))


def prep_chunk(src, dst):
    """Raw counts in the order of the 19,264 scFoundation genes (missing genes = 0)."""
    genes = pd.read_csv(SCF / "OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"].tolist()
    a = ad.read_h5ad(src)
    names = list(a.var["gene_name"] if "gene_name" in a.var.columns else a.var_names)
    pos = {g: i for i, g in enumerate(genes)}
    rows = [i for i, g in enumerate(names) if g in pos]
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, [pos[names[i]] for i in rows])),
                      shape=(len(names), len(genes)))
    X = (sp.csr_matrix(a.X) @ M).tocsr().astype(np.float32)
    ad.AnnData(X=X, obs=pd.DataFrame(index=a.obs_names), var=pd.DataFrame(index=genes)).write_h5ad(dst)


def run(gpu, part, nparts):
    patch_load_py()
    for d in (OUT_DIR, PREP_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # cells of varying length fragment the CUDA cache
    for f in chunks()[part::nparts]:
        if result(f).exists():
            continue
        prep = PREP_DIR / f.name
        prep_chunk(f, prep)
        with open(LOG_DIR / f"{f.stem}.log", "w") as log:
            subprocess.run([sys.executable, "get_embedding.py", "--task_name", f.stem, "--input_type", "singlecell",
                            "--output_type", "cell", "--pool_type", "all", "--tgthighres", "t4",
                            "--data_path", str(prep), "--save_path", f"{OUT_DIR}/", "--pre_normalized", "F",
                            "--version", "ce"], cwd=SCF, env=env, stdout=log, stderr=subprocess.STDOUT)
        prep.unlink()
        if not result(f).exists():
            sys.exit(f"FAILED {f.stem}: see {LOG_DIR / f'{f.stem}.log'}")
        print(f"done {f.stem}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["run", "merge"])
p.add_argument("--gpu", default="0")
p.add_argument("--part", type=int, default=0)
p.add_argument("--nparts", type=int, default=1)
p.add_argument("--out", help="output .npy (default scfm.embeddings.scFoundation)")
a = p.parse_args()
if a.cmd == "run":
    run(a.gpu, a.part, a.nparts)
else:
    missing = [f.stem for f in chunks() if not result(f).exists()]
    if missing:
        sys.exit(f"unfinished chunks: {missing}")
    merge(((read_obs_names(f), np.load(result(f))) for f in chunks()), output("scFoundation", a.out))  # chunk cell order
