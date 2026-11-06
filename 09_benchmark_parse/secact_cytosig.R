# SecAct and CytoSig on the Parse-PBMC benchmark (SecAct environment, R >= 4.2).
# Activity z-scores from the pseudobulk log fold changes (prepare_inputs.py): SecAct signatures
# (SecAct.activity.inference defaults) and CytoSig signatures (sigMatrix = "CytoSig", lambda = 10000, signatures
# scored separately). Per condition the signatures mapped to CYTAP cytokines (R/ligand_maps.R) are ranked by z;
# true_rank = best rank of the cytokine's signatures. The cytokine -> signature maps ({SecAct,CytoSig}_map_raw.csv)
# are also used by 10_benchmark_in_vivo.
# Output: {SecAct,CytoSig}_output/*_activity.rds, *_top10_results.csv, *_ligands.txt (ranking pool)
#   Rscript 09_benchmark_parse/secact_cytosig.R
source("R/paths.R")
source("R/ligand_maps.R")
suppressPackageStartupMessages({ library(tidyverse); library(SecAct) })

logfc <- as.matrix(read.csv(bench_path("parse_dir", "SecAct_input", "pseudobulk_log2fc.csv"),
                            row.names = 1, check.names = FALSE))
map_raw <- read_csv(cfg_path("reference", "cytokine_ligand_map"), show_col_types = FALSE)
responsive <- read_csv(bench_path("parse_dir", "benchmark", "responsive_conditions.csv"), show_col_types = FALSE)

run_method <- function(method) {
  act <- if (method == "SecAct") SecAct.activity.inference(inputProfile = logfc, is.differential = TRUE) else
    SecAct.activity.inference(inputProfile = logfc, is.differential = TRUE, sigMatrix = "CytoSig", lambda = 10000,
                              is.group.sig = FALSE)
  dir <- paste0(method, "_output")
  saveRDS(act, new_file(bench_path("parse_dir", dir, paste0(tolower(method), "_activity.rds"))))
  zmat <- act$zscore
  mp_raw <- if (method == "SecAct") secact_map(map_raw, rownames(zmat)) else cytosig_map(map_raw, rownames(zmat))
  write_csv(mp_raw, new_file(bench_path("parse_dir", paste0(method, "_map_raw.csv"))))
  mp <- mp_raw %>% filter(!is.na(.data[[method]]), .data[[method]] != "")
  cyt_to_lig <- setNames(mp[[method]], mp$Cytokines)
  mapped <- unique(unlist(strsplit(mp[[method]], ";")))
  writeLines(mapped, new_file(bench_path("parse_dir", dir, paste0(tolower(method), "_ligands.txt"))))
  res <- bind_rows(lapply(seq_len(nrow(responsive)), function(i)
    score_mapped(zmat, responsive$celltype[i], responsive$cytokine[i], cyt_to_lig, mapped)))
  write_csv(res, new_file(bench_path("parse_dir", dir, paste0(tolower(method), "_top10_results.csv"))))
  print(count(res, status))
}
run_method("SecAct")
run_method("CytoSig")
