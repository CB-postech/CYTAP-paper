# ContactTracing on the in vivo data: rank of the target receptor (CSF3R; IL7R) among receptors with >= 1 significant
# response gene, in the full list and among the receptors of the CYTAP cytokines (MAP).
# Output: results/benchmark_in_vivo_contacttracing_summary.csv
#   Rscript 10_benchmark_in_vivo/contacttracing_rank.R
source("R/paths.R")
suppressPackageStartupMessages({ library(tidyverse); library(readxl) })
vdir <- function(...) file.path(cfg_path("benchmark", "in_vivo_dir"), ...)
cytap_receptors <- read_excel(cfg_path("reference", "cytokine_receptors")) %>%
  select(receptor = `Receptor gene symbol`) %>% separate_rows(receptor, sep = "[;,]") %>%
  mutate(receptor = trimws(receptor)) %>% filter(!is.na(receptor), receptor != "") %>% pull(receptor) %>% unique()

ct_rank <- function(run) read.delim(vdir("ContactTracing_output", run, "receptor_activity_ranking.txt"),
                                    sep = "\t", row.names = 1, check.names = FALSE)

ct_summary <- function(run, ct, target) {
  d <- ct_rank(run) %>% filter(cell_type_receptor == ct)
  bind_rows(lapply(c("full", "MAP"), function(lab) {
    s <- if (lab == "MAP") filter(d, receptor %in% cytap_receptors) else d
    sig <- s %>% filter(numSigI1_fdr05 > 0) %>% mutate(rank = rank(-numSigI1_fdr05, ties.method = "max"))
    r <- sig$rank[sig$receptor == target]
    tibble(dataset = run, celltype = ct, list = lab, target = target, tested = nrow(s), calls = nrow(sig),
           rank = if (length(r)) r else NA_integer_)
  }))
}
ct_tab <- bind_rows(ct_summary("GCSF_BM_mye", "Mye_lineage", "CSF3R"), ct_summary("IL7_T", "CD4_T", "IL7R"),
                    ct_summary("IL7_T", "CD8_T", "IL7R"))
print(ct_tab)
write_csv(ct_tab, file.path(work_dir, cfg$results, "benchmark_in_vivo_contacttracing_summary.csv"))
