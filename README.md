# CYTAP-paper

Analysis code of *CYTAP: inferring cytokine responses in single cells from adapted foundation model embeddings*
(Jeong & Kim, 2026; journal and DOI to be added). The method is the [`cytap`](https://github.com/CB-postech/CYTAP)
package; data, models and result tables are on Zenodo (DOI to be added).

## Setup

```bash
git clone https://github.com/CB-postech/CYTAP-paper.git && cd CYTAP-paper
pip install git+https://github.com/CB-postech/CYTAP.git scanpy gseapy h5py patsy openpyxl pyyaml
```

R ≥ 4.2: Seurat, SeuratDisk, harmony, tidyverse, readxl, yaml. State, CellOT, the
other foundation models and the benchmarked methods run in their own environments (see the script headers).
Paths are set in `config/paths.yaml` (relative to the repository root); folders of external tools go in
`config/paths.local.yaml` (template: `config/paths.local.example.yaml`). Run all scripts from the repository root.
Each step writes its result tables to `results/`; the tables behind the figures are on Zenodo.

## Analyses

| Step | Methods section | Scripts | Output (`results/`) |
|---|---|---|---|
| 00 | Data collection and pre-processing; cell embeddings | `00_preprocessing/`: `preprocess.R`, `preprocess.py`, `state_embeddings.sh` | State embeddings (`data/`) |
| 01 | Model training | `01_training/train_cytap.py` | `models/` |
| 02 | Inference (paired, unpaired) | `02_inference/run_cytap_inference.sh` | `inference/` |
| 03 | Statistical analysis and effect size; response signature scores | `03_query_analysis/query_analysis.py` | `ifnb/`, `macrophage/`, `il7/`, `gcsf/` |
| 04 | Correlation with disease severity; Luminex reanalysis (COMBAT) | `04_combat/combat.py` | `combat/` |
| 05 | Counterfactual control reference (CellOT); evaluation of control references | `05_cellot/`: `cellot_maps.py`, `train_cellot.sh`, `eval_control_references.py` | `cellot/` |
| 06 | Donor-held-out input ablation; full-pipeline test (macrophages) | `06_input_ablation/input_ablation.py` | `input_ablation/` |
| 07 | Variance partitioning of input representations | `07_variance_partitioning/variance_partitioning.py` | `variance_partitioning/` |
| 08 | Comparison of pretrained cell embeddings | `08_embedding_comparison/`: `embed_*.py`, `cv_embeddings.py` | `scfm_embeddings/` |
| 09 | Benchmarking on Parse-PBMC | `09_benchmark_parse/`: `prepare_inputs.py`, one script per method, `recall_summary.R` | `benchmark_parse/` |
| 10 | Benchmarking on in vivo data (incl. Scupa) | `10_benchmark_in_vivo/`: one script per method, `contacttracing_rank.R` | `benchmark_in_vivo/` |

Order: 05 (COMBAT maps) before the COMBAT CellOT run of 02; the cross-validation models of 06 before
`05_cellot/eval_control_references.py` and `09_benchmark_parse/cytap_oof.py`. Usage and settings are in each
script's header.

## Data

| Dataset | Source | Used in |
|---|---|---|
| Parse 10M PBMC cytokine screen, training data and DEG table (Oesinghaus et al., bioRxiv 2025) | [Parse Biosciences](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-exp) | 00, 01, 05–09 |
| IFN-β-stimulated PBMCs (Kang et al., Nat. Biotechnol. 2018) | GEO GSE96583 | 00, 02, 03 |
| Cytokine-stimulated macrophages (Zhang et al., Genome Med. 2021) | GEO GSE168710 | 00, 02, 03, 06 |
| G-CSF-primed bone marrow (You et al., Cell Discov. 2022) | GEO GSE193138 | 00, 02, 03, 10 |
| rhIL-7-hyFc-treated patients, T cells (Jang et al., JCI Insight 2026) | Zenodo 10.5281/zenodo.17958065 | 00, 02, 03, 10 |
| COMBAT CITE-seq (COMBAT Consortium, Cell 2022) | CZ CELLxGENE Discover | 00, 02, 04, 05 |
| COMBAT plasma Luminex (COMBAT Consortium, Cell 2022) | Zenodo 10.5281/zenodo.6120249 (`Oxford data_output_combined.xlsx` → `data/query/combat/`) | 04 |
| Immune Dictionary signatures (IREA), Human Cytokine Dictionary (huCIRA), NicheNet prior (Zenodo 7074291) | original publications | 09, 10 |

The State SE-600M model ([Arc Institute](https://github.com/ArcInstitute/state)) gives the cell embeddings
(`state emb transform --model-folder SE-600M`).

## Zenodo record

Unpack in the repository root (`tar -xf <file>`):

| File | Content |
|---|---|
| `models.tar` | CYTAP model (`models/`) and COMBAT CellOT maps (`results/cellot/`); both also downloaded automatically by `cytap` |
| `query_slim.tar` | query datasets with cell metadata and State embeddings, no expression (`data/query_slim/`) |
| `inference_slim.tar` | CYTAP predictions (`results/inference_slim/`) |
| `results.tar.gz` | result tables behind the figures, incl. the benchmark outputs (`results/`); curated tables (`data/reference/`); State licence files |

With these files the result tables can be checked directly, and CYTAP can be rerun on the query datasets (set
`data.*` in `config/paths.yaml` to `data/query_slim/`). Analyses that need gene expression start from the original datasets.

## License

Code: MIT (`LICENSE`). Model weights, State embeddings and CYTAP predictions: CC BY-NC 4.0 and the Arc Institute
State Model Non-Commercial License. The datasets and the benchmarked tools are not redistributed; cite their
original publications.
