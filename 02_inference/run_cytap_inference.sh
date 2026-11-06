#!/usr/bin/env bash
# CYTAP predictions for the query datasets (paths from config/paths.yaml). Run from the repository root.
# COMBAT with CellOT controls needs the map of 05_cellot (results/cellot/combat_cellot.pt, also on Zenodo).
# Existing outputs are kept (delete them to rerun).
set -euo pipefail
P() { python -c "import sys; sys.path.insert(0, '.'); from paper_utils import path; print(path(*sys.argv[1:]))" "$@"; }
MODEL=$(P model)
mkdir -p "$(dirname "$(P inference ifnb)")"
predict() { if [[ -e "$2" ]]; then echo "skip (exists): $2"; else cytap predict "$@" --model "$MODEL"; fi; }

# paired: each donor's own control cells
predict "$(P data ifnb)" "$(P inference ifnb)" \
    --design paired --control-key stim --control-label ctrl --donor-key ind --celltype-key cell
predict "$(P data macrophage)" "$(P inference macrophage)" \
    --design paired --control-key condition --control-label Control --donor-key sample
predict "$(P data gcsf)" "$(P inference gcsf)" \
    --design paired --control-key condition --control-label pre --donor-key donor --celltype-key celltype
predict "$(P data il7)" "$(P inference il7)" \
    --design paired --control-key treatment --control-label baseline --donor-key patient --celltype-key celltype

# unpaired (COMBAT): pooled healthy-donor controls, and CellOT counterfactual controls
predict "$(P data combat)" "$(P inference combat_pooled)" \
    --design unpaired --control-key Source --control-label HV --celltype-key major_subset
predict "$(P data combat)" "$(P inference combat_cellot)" \
    --design unpaired --reference counterfactual --counterfactual "$(P cellot maps)/combat_cellot.pt" \
    --control-key Source --control-label HV --donor-key donor_id --celltype-key major_subset
