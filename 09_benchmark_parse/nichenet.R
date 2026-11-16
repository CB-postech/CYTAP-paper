# NicheNet on the Parse-PBMC benchmark.
# Per responsive condition: geneset = DEGs (adj. p < 0.05, |log FC| >= 0.25) of the cell type, background =
# genes detected in >= 5% of its cells, both restricted to the ligand-target matrix (human, Zenodo 7074291).
# Candidate ligands = ligands of the CYTAP cytokines (cytokine_ligand_map.csv; IL8 -> CXCL8) present in
# the prior. Ligands ranked by corrected AUPR; true_rank = best rank of the cytokine's ligands.
# Conditions with < 10 genes after filtering are not scored ("no_geneset", true_rank NA; not evaluable in recall_summary.R).
# Output: NicheNet_output/nichenet_top10_results.csv, candidate_ligands.txt (ranking pool)
#   Rscript 09_benchmark_parse/nichenet.R
source("R/paths.R")
suppressPackageStartupMessages({ library(nichenetr); library(tidyverse) })

LOGFC_CUT <- 0.25; PADJ_CUT <- 0.05; RUN_MIN_GENES <- 10

ligand_target <- nichenet_prior("ligand_target_matrix_nsga2r_final.rds")
deg <- read_csv(bench_path("parse_dir", "deg_mapped.csv"), show_col_types = FALSE)
bg  <- read_csv(bench_path("parse_dir", "background_by_celltype.csv"), show_col_types = FALSE)
responsive <- read_csv(bench_path("parse_dir", "benchmark", "responsive_conditions.csv"), show_col_types = FALSE)

map_long <- read_csv(cfg_path("reference", "cytokine_ligand_map"), show_col_types = FALSE) %>%
  separate_rows(ligand_symbols, sep = ",\\s*") %>%
  mutate(ligand = if_else(ligand_symbols == "IL8", "CXCL8", ligand_symbols),
         in_prior = ligand %in% colnames(ligand_target))
candidate_ligands <- map_long %>% filter(in_prior) %>% pull(ligand) %>% unique()
writeLines(candidate_ligands, new_file(bench_path("parse_dir", "NicheNet_output", "candidate_ligands.txt")))

score_condition <- function(la, true_cytokine, top_n = 10) {
  tl <- map_long %>% filter(Cytokines == true_cytokine, in_prior) %>% pull(ligand)
  if (length(tl) == 0) return(tibble(true_rank = NA_integer_, hit = NA))  # no ligand in the prior
  ranks <- la$rank[la$test_ligand %in% tl]
  best <- if (length(ranks)) min(ranks) else Inf
  tibble(true_rank = ifelse(is.finite(best), best, NA_integer_), hit = is.finite(best) && best <= top_n)
}

results <- vector("list", nrow(responsive))
for (i in seq_len(nrow(responsive))) {
  ct <- responsive$celltype[i]; cyt <- responsive$cytokine[i]
  background <- bg %>% filter(celltype == ct) %>% pull(gene) %>% intersect(rownames(ligand_target))
  geneset <- deg %>%
    filter(celltype_adata == ct, cytokine == cyt, adj_p_value < PADJ_CUT, abs(log_fc) >= LOGFC_CUT) %>%
    pull(gene) %>% intersect(rownames(ligand_target)) %>% intersect(background)
  if (length(background) == 0 || length(geneset) < RUN_MIN_GENES) {
    results[[i]] <- tibble(celltype = ct, cytokine = cyt, true_rank = NA_integer_, hit = FALSE, status = "no_geneset")
    next
  }
  la <- predict_ligand_activities(geneset = geneset, background_expressed_genes = background,
                                  ligand_target_matrix = ligand_target, potential_ligands = candidate_ligands) %>%
    arrange(desc(aupr_corrected)) %>% mutate(rank = row_number())
  sc <- score_condition(la, cyt)
  results[[i]] <- tibble(celltype = ct, cytokine = cyt, true_rank = sc$true_rank, hit = sc$hit,
                         status = if (is.na(sc$hit)) "no_prior_ligand" else "scored")
}
res <- bind_rows(results)
write_csv(res, new_file(bench_path("parse_dir", "NicheNet_output", "nichenet_top10_results.csv")))
print(count(res, status))
print(res %>% summarise(n = n(), top10_scored = mean(true_rank <= 10, na.rm = TRUE),
                        top10_all = mean(coalesce(true_rank <= 10, FALSE))))
