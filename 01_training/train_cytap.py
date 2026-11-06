"""Train the CYTAP classifier on Parse-PBMC (all 12 donors) with cytap.train.

comp: the CYTAP model (config: model). raw / adj / dev / hvg: models with the other inputs of the input
ablation, used in the macrophage pipeline test (config: ablation_models). Training settings are the cytap.train
defaults (MLP 2,048-1,024, dropout 0.3, focal loss gamma 2, AdamW + one-cycle schedule, 10 epochs).
HVG input: 00_preprocessing/preprocess.py parse-hvg.

python 01_training/train_cytap.py --inputs comp
python 01_training/train_cytap.py --inputs raw adj dev hvg
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cytap.train import train_classifier
from paper_utils import load_hvg, load_parse, path, reference_inputs

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", choices=["comp", "raw", "adj", "dev", "hvg"], required=True)
args = p.parse_args()


def target(rep):
    f = path("model") if rep == "comp" else path("ablation_models", rep)
    if f.exists():
        sys.exit(f"{f} exists")
    f.parent.mkdir(parents=True, exist_ok=True)
    return str(f)


emb = [r for r in args.inputs if r != "hvg"]
if emb:
    Z, obs = load_parse()
    for rep in emb:
        X, y, classes, _, _ = reference_inputs(Z, obs["cytokine"].to_numpy(), obs["donor"].to_numpy(),
                                               obs["cell_type"].to_numpy(), representation=rep)
        print(f"{rep}: training input {X.shape}, {len(classes)} classes")
        train_classifier(X, y, classes, save_path=target(rep), config={"representation": rep, "embedding_key": "X_state"})
        del X
        gc.collect()

if "hvg" in args.inputs:
    X, y, _, _, classes = load_hvg()
    genes = [g.strip() for g in open(path("parse", "hvg_genes")) if g.strip()]
    train_classifier(X, y, classes, save_path=target("hvg"),
                     config={"representation": "hvg", "genes": genes, "embedding_key": "", "embedding_model": ""})
