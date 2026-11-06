# NicheNet on the in vivo datasets (receiver = cell type of interest, treated vs baseline).
# Geneset: FindMarkers (min.pct 0.05) adj. p < 0.05 and |avg log2FC| >= 0.25, in the ligand-target matrix.
#   full: potential ligands = ligands of receptors detected in the receiver (pct = 0), background = genes
#         detected in the receiver in the ligand-target matrix
#   MAP : potential ligands = ligands of the CYTAP cytokines (cytokine_ligand_map.csv; IL8 -> CXCL8),
#         background = all genes of the receiver object
# Ligands with pearson = NA (constant target-potential vector) are removed.
# Output (benchmark.in_vivo_dir/NicheNet): {GCSF_Mye, IL7_combined_CD4T, IL7_combined_CD8T}_ligand_activities_f.csv (full)
#                                          and *_parse_ligand_activities_f.csv (MAP)
#   Rscript 10_benchmark_in_vivo/nichenet.R
source("R/paths.R")
suppressPackageStartupMessages({ library(nichenetr); library(Seurat); library(tidyverse) })

ligand_target <- nichenet_prior("ligand_target_matrix_nsga2r_final.rds")
lr_network    <- nichenet_prior("lr_network_human_21122021.rds")
map_long <- read_csv(cfg_path("reference", "cytokine_ligand_map"), show_col_types = FALSE) %>%
  separate_rows(ligand_symbols, sep = ",\\s*") %>%
  mutate(ligand = if_else(ligand_symbols == "IL8", "CXCL8", ligand_symbols))
cytap_ligands <- map_long %>% filter(ligand %in% colnames(ligand_target)) %>% pull(ligand) %>% unique()

run_nichenet <- function(obj, receiver, group_by, case, ctrl, name) {
  Idents(obj) <- "celltype"
  expressed <- get_expressed_genes(receiver, obj, pct = 0)
  receptors <- intersect(unique(lr_network$to), expressed)
  potential <- lr_network %>% filter(to %in% receptors) %>% pull(from) %>% unique()
  recv <- subset(obj, idents = receiver)
  de <- FindMarkers(recv, ident.1 = case, ident.2 = ctrl, group.by = group_by, min.pct = 0.05) %>%
    rownames_to_column("gene")
  geneset <- de %>% filter(p_val_adj < 0.05 & abs(avg_log2FC) >= 0.25) %>% pull(gene) %>%
    intersect(rownames(ligand_target))
  message(sprintf("%s: %d genes in geneset", name, length(geneset)))
  run <- function(ligands, background, suffix) {
    la <- predict_ligand_activities(geneset = geneset, background_expressed_genes = background,
                                    ligand_target_matrix = ligand_target, potential_ligands = ligands) %>%
      arrange(-aupr_corrected) %>% mutate(rank = rank(desc(aupr_corrected))) %>% filter(!is.na(pearson))
    write.csv(la, new_file(bench_path("in_vivo_dir", "NicheNet", paste0(name, suffix, "_ligand_activities_f.csv"))))
  }
  run(potential, intersect(expressed, rownames(ligand_target)), "")
  run(cytap_ligands, rownames(recv), "_parse")
}

bm <- readRDS(cfg_path("data", "gcsf_seurat"))
run_nichenet(bm, "Mye_lineage", "condition", "post", "pre", "GCSF_Mye")

il7 <- readRDS(cfg_path("data", "il7_seurat"))
for (ct in c("CD4_T", "CD8_T"))
  run_nichenet(il7, ct, "treatment", "post_dose1_wk1", "baseline", paste0("IL7_combined_", sub("_", "", ct)))
