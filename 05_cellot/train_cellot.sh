#!/usr/bin/env bash
# Train the CellOT maps (Bunne et al. 2023) in the CellOT environment, from the repository root.
#   Parse-PBMC: cytokine -> PBS, one map per CV fold (that fold's held-out donors) and one on all donors.
#   COMBAT:     COVID_MILD / COVID_SEV / COVID_CRIT -> HV.
# Setup: git clone https://github.com/bunnech/cellot (commit 522d2b9) and apply 05_cellot/cellot_gpu.patch in it
# (device placement only); CELLOT = that folder. Inputs: cellot_maps.py prepare. Settings: 05_cellot/configs/
# (CellOT example configuration, 10,000 iterations). Then: cellot_maps.py export.
# PARSE_RUNS / COMBAT=0 select runs, e.g. training-donor targets:
#   PARSE_RUNS="parsePBMC_f1_trainctrl parsePBMC_f2_trainctrl parsePBMC_f3_trainctrl parsePBMC_f4_trainctrl" COMBAT=0
set -euo pipefail
P() { python -c "import sys; sys.path.insert(0, '.'); from paper_utils import path; print(path(*sys.argv[1:]))" "$@"; }
CFG=$PWD/05_cellot/configs
DATA=$(P cellot datasets)
OUT=$(P cellot results)
CELLOT=${CELLOT:-$HOME/cellot}
cd "$CELLOT"

PARSE_RUNS=${PARSE_RUNS:-"parsePBMC_f1 parsePBMC_f2 parsePBMC_f3 parsePBMC_f4 parsePBMC_nofold"}
for run in $PARSE_RUNS; do
  python scripts/train.py --outdir "$OUT/$run/cyt_to_PBS_pc100" \
    --config "$CFG/parsePBMC_task.yaml" --config "$CFG/cellot.yaml" \
    --config.data.path "$DATA/${run}_data_pca100.h5ad" --config.data.source cytokine
done

for src in $([ "${COMBAT:-1}" = 1 ] && echo MILD SEV CRIT); do
  lc=$(echo $src | tr A-Z a-z)
  python scripts/train.py --outdir "$OUT/combat/covid_${lc}_to_hv_pc100" \
    --config "$CFG/combat_task.yaml" --config "$CFG/cellot.yaml" \
    --config.data.path "$DATA/combat_pca100.h5ad" --config.data.source COVID_$src
done
