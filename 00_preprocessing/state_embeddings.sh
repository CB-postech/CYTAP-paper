#!/usr/bin/env bash
# State SE-600M embeddings (obsm['X_state']) of the log-normalized data, in the State environment
# (https://github.com/ArcInstitute/state). Run from the repository root; existing outputs are kept.
#   SE600M : SE-600M model folder (default ./SE-600M, Hugging Face arcinstitute/SE-600M)
set -euo pipefail
P() { python -c "import sys; sys.path.insert(0, '.'); from paper_utils import path; print(path(*sys.argv[1:]))" "$@"; }
SE600M=${SE600M:-SE-600M}
IN=data/normalized
embed() { if [[ -e "$2" ]]; then echo "skip (exists): $2"; else state emb transform --model-folder "$SE600M" --input "$1" --output "$2"; fi; }

embed "$(P parse normalized)" "$(P parse emb)"                                # Parse-PBMC (training data)
embed "$IN/IFNb_data_total.h5ad" "$(P data ifnb)"                             # preprocess.R ifnb
embed "$IN/macrophage_stim_normalized.h5ad" "$(P data macrophage)"            # preprocess.R / preprocess.py macrophage, normalize
embed "$IN/bm_normalized.h5ad" "$(P data gcsf)"                               # preprocess.R gcsf, preprocess.py normalize --counts raw
for c in CRC GBM; do   # IL-7 per cohort (preprocess.py normalize); annotated T cells of both combined -> data.il7
  embed "$IN/IL7_tcell_$c.h5ad" "data/query/IL7_Tcell/IL7_tcell_${c}_state_emb.h5ad"
done
embed "$(P data combat_expression_unfiltered)" "$(P data combat_unfiltered)"  # preprocess.py combat (CELLxGENE X is log-normalized)
