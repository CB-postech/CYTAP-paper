# SecAct and CytoSig on the in vivo datasets (SecAct environment, R >= 4.2).
#   SecAct : SecAct.CCC.scRNAseq (treated vs baseline per cell type) -> SecretedProteinActivity
#   CytoSig: same pseudobulk differential profile as SecAct (counts summed per cell type and condition
#            -> CPM -> log2(x + 1), treated - baseline; >= 30 cells per group), CytoSig signatures via
#            SecAct.activity.inference(sigMatrix = "CytoSig", lambda = 10000)
# Per cell type of interest, signatures are ranked by z-score (ties: min rank) with BH-FDR over all
# signatures; the CYTAP column gives the CYTAP cytokine(s) mapped to each signature (rank_activity, R/ligand_maps.R).
# Output (benchmark.in_vivo_dir): SecAct/SecAct_{G-CSF_treated_BM,IL7_treated_Tcell}.rds,
#   CytoSig/CytoSig_{G-CSF_treated_BM,IL7_treated_T}.rds and {method}_{GCSF_Mye_lineage,IL7_CD4_T,IL7_CD8_T}_ranking.csv
#   (ranking.csv: rank over all signatures and mapped_rank over the signatures mapped to CYTAP cytokines)
#   Rscript 10_benchmark_in_vivo/secact_cytosig.R
source("R/paths.R")
source("R/ligand_maps.R")
suppressPackageStartupMessages({ library(Seurat); library(SecAct); library(Matrix); library(tidyverse) })

map_raw <- read_csv(cfg_path("reference", "cytokine_ligand_map"), show_col_types = FALSE)

# pseudobulk log2 CPM difference per cell type (SecAct.CCC.scRNAseq step 2)
pseudobulk_diff <- function(obj, cond_col, case, ctrl) {
  counts <- SecAct:::extract_seurat_counts(obj)
  rownames(counts) <- SecAct:::transferSymbol(rownames(counts))
  counts <- SecAct:::rm_duplicates(counts)
  meta <- obj@meta.data
  pb <- function(m) { v <- Matrix::rowSums(m); v <- v * 1e6 / sum(v); log2(v + 1) }
  cts <- intersect(meta[meta[[cond_col]] == case, "celltype"], meta[meta[[cond_col]] == ctrl, "celltype"])
  diff <- data.frame()
  for (ct in unique(cts)) {
    m_case <- counts[, meta[[cond_col]] == case & meta$celltype == ct, drop = FALSE]
    m_ctrl <- counts[, meta[[cond_col]] == ctrl & meta$celltype == ct, drop = FALSE]
    if (ncol(m_case) < 30 || ncol(m_ctrl) < 30) next
    d <- pb(m_case) - pb(m_ctrl)
    diff[names(d), ct] <- d
  }
  diff
}

run_dataset <- function(obj, cond_col, case, ctrl, sec_rds, cs_rds, tag, celltypes) {
  obj <- SecAct.CCC.scRNAseq(obj, cellType_meta = "celltype", condition_meta = cond_col,
                             conditionCase = case, conditionControl = ctrl)
  sec <- obj@misc$SecAct_output$SecretedProteinActivity
  saveRDS(sec, new_file(bench_path("in_vivo_dir", "SecAct", sec_rds)))
  cs <- SecAct.activity.inference(inputProfile = pseudobulk_diff(obj, cond_col, case, ctrl), is.differential = TRUE,
                                  sigMatrix = "CytoSig", lambda = 10000, is.group.sig = FALSE, nrand = 1000)
  saveRDS(cs, new_file(bench_path("in_vivo_dir", "CytoSig", cs_rds)))
  sec_map <- secact_map(map_raw, rownames(sec$zscore))
  cs_map <- cytosig_map(map_raw, rownames(cs$zscore))
  for (ct in celltypes) {
    write_csv(rank_activity(sec, ct, "SecAct", sec_map),
              new_file(bench_path("in_vivo_dir", "SecAct", sprintf("SecAct_%s_%s_ranking.csv", tag, ct))))
    if (ct %in% colnames(cs$zscore))
      write_csv(rank_activity(cs, ct, "CytoSig", cs_map),
                new_file(bench_path("in_vivo_dir", "CytoSig", sprintf("CytoSig_%s_%s_ranking.csv", tag, ct))))
  }
}

run_dataset(readRDS(cfg_path("data", "gcsf_seurat")), "condition", "post", "pre",
            "SecAct_G-CSF_treated_BM.rds", "CytoSig_G-CSF_treated_BM.rds", "GCSF", "Mye_lineage")
run_dataset(readRDS(cfg_path("data", "il7_seurat")), "treatment", "post_dose1_wk1", "baseline",
            "SecAct_IL7_treated_Tcell.rds", "CytoSig_IL7_treated_T.rds", "IL7", c("CD4_T", "CD8_T"))
