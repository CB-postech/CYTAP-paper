# Scupa polarization scores on UCE embeddings (supplementary figure).
# UCE embeddings first (UCE environment, in the UCE repository; 4-layer model, default cell filtering), from raw
# counts in .X (G-CSF: data.gcsf_counts; IL-7: layers['counts'] of data.il7_counts) -> data.uce_gcsf, data.uce_il7:
#   python eval_single_anndata.py --adata_path bm.h5ad --dir ./uce_output/ --species human --batch_size=50
# Baseline cells are given as unpolarized reference (MeasurePolar(unpolarized_cell = ...)).
#   G-CSF Mye_lineage: Scupa cell type Mono (used in the paper);  IL-7: CD4T (CD4_T cells), CD8T (CD8_T cells)
# Output (benchmark.in_vivo_dir/Scupa): cell metadata with the polarization scores (*_score columns),
#   BM_seu_mye_Mono_meta.csv, IL7_seu_{cd4,cd8}_meta.csv
#   Rscript 10_benchmark_in_vivo/scupa.R
source("R/paths.R")
suppressPackageStartupMessages({ library(Seurat); library(Scupa) })

load_uce <- function(f) {
  obj <- schard::h5ad2seurat(f)
  obj[["uce"]] <- CreateAssayObject(data = t(Embeddings(obj, "Xuce_")))
  DefaultAssay(obj) <- "uce"
  VariableFeatures(obj) <- rownames(obj)
  obj
}
save <- function(obj, name) write.csv(obj@meta.data, new_file(bench_path("in_vivo_dir", "Scupa", name)))

bm <- load_uce(cfg_path("data", "uce_gcsf"))
mye <- subset(bm, subset = celltype == "Mye_lineage")
base <- colnames(mye)[mye$condition == "pre"]
save(MeasurePolar(mye, celltype = "Mono", embedding = "uce", unpolarized_cell = base), "BM_seu_mye_Mono_meta.csv")

il7 <- load_uce(cfg_path("data", "uce_il7"))
for (x in list(c("CD4_T", "CD4T", "cd4"), c("CD8_T", "CD8T", "cd8"))) {
  s <- subset(il7, subset = celltype == x[1])
  save(MeasurePolar(s, celltype = x[2], embedding = "uce", unpolarized_cell = colnames(s)[s$treatment == "baseline"]),
       paste0("IL7_seu_", x[3], "_meta.csv"))
}
