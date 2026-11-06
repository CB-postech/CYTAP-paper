# Parse-PBMC benchmark summary: top-k recall of the true cytokine per responsive condition (input: method outputs in
# benchmark.parse_dir). Recall uses the conditions each method can score; the random level is the mean over conditions
# of k / pool size (fixed pool: CYTAP classes, NicheNet candidate ligands, SecAct / CytoSig mapped signatures;
# per-condition pool: huCIRA, IREA, ContactTracing); normalized = (recall - random) / (1 - random).
# Output: results/benchmark_parse_summary.csv (top-5 / top-10 recall per method), benchmark_parse_recall_curve.csv (k = 1-20)
#   Rscript 09_benchmark_parse/recall_summary.R
source("R/paths.R")
suppressPackageStartupMessages(library(tidyverse))

pdir <- function(...) file.path(cfg_path("benchmark", "parse_dir"), ...)
METHODS <- c("CYTAP", "huCIRA", "NicheNet", "SecAct", "IREA", "CytoSig", "ContactTracing")
K <- 1:20

count_lines <- function(p) if (file.exists(p)) length(readLines(p)) else NA_integer_
FIXED_POOL <- c(CYTAP    = count_lines(pdir("CYTAP_output", "cytap_classes.txt")),
                NicheNet = count_lines(pdir("NicheNet_output", "candidate_ligands.txt")),
                SecAct   = count_lines(pdir("SecAct_output", "secact_ligands.txt")),
                CytoSig  = count_lines(pdir("CytoSig_output", "cytosig_ligands.txt")))

load_one <- function(p, nm, pool_path = NULL) {
  if (!file.exists(p)) { message("[skip] not found: ", p); return(NULL) }
  d <- read_csv(p, show_col_types = FALSE)
  out <- d %>% select(celltype, cytokine, true_rank)
  out$pool_size <- if ("pool_size" %in% names(d)) d$pool_size else NA_real_
  if (!is.null(pool_path)) {
    pl <- read_csv(pool_path, show_col_types = FALSE) %>% select(celltype, cytokine, pool_size)
    out <- out %>% select(-pool_size) %>% left_join(pl, by = c("celltype", "cytokine"))
  }
  if (nm %in% names(FIXED_POOL) && all(is.na(out$pool_size))) out$pool_size <- FIXED_POOL[[nm]]
  out$method <- nm
  out
}
dat <- bind_rows(
  load_one(pdir("NicheNet_output", "nichenet_top10_results.csv"), "NicheNet"),
  load_one(pdir("CYTAP_output", "cytap_top10_results.csv"), "CYTAP"),
  load_one(pdir("huCIRA_output", "hucira_top10_results_default.csv"), "huCIRA"),
  load_one(pdir("SecAct_output", "secact_top10_results.csv"), "SecAct"),
  load_one(pdir("IREA_output", "irea_top10_results.csv"), "IREA"),
  load_one(pdir("CytoSig_output", "cytosig_top10_results.csv"), "CytoSig"),
  load_one(pdir("ContactTracing_output", "contacttracing_true_rank.csv"), "ContactTracing",
           pdir("ContactTracing_output", "ct_pool_by_condition.csv")))
stopifnot("no result files found" = nrow(dat) > 0)
all_pairs <- dat %>% distinct(celltype, cytokine)
tot_cyt <- n_distinct(all_pairs$cytokine)

# recall and random level per k over the evaluable conditions of each method
curves <- bind_rows(lapply(intersect(METHODS, unique(dat$method)), function(nm) {
  ev <- dat %>% filter(method == nm, !is.na(true_rank))
  if (nrow(ev) == 0) return(NULL)
  tibble(series = nm, x = K, recall = sapply(K, function(k) mean(ev$true_rank <= k)),
         rand = sapply(K, function(k) mean(pmin(k / ev$pool_size, 1), na.rm = TRUE)),
         n_ev = nrow(ev), n_cyt = n_distinct(ev$cytokine), pool_mean = mean(ev$pool_size, na.rm = TRUE),
         pool_min = min(ev$pool_size, na.rm = TRUE), pool_max = max(ev$pool_size, na.rm = TRUE))
}))
at <- function(v, x, k) v[which(x == k)[1]]
tab <- curves %>% group_by(series) %>%
  summarise(n = first(n_ev), n_cyt = first(n_cyt),
            pool = ifelse(first(pool_min) == first(pool_max), sprintf("%.0f", first(pool_mean)),
                          sprintf("%.0f-%.0f (mean %.0f)", first(pool_min), first(pool_max), first(pool_mean))),
            recall5 = at(recall, x, 5), rand5 = at(rand, x, 5), norm5 = (recall5 - rand5) / (1 - rand5),
            recall10 = at(recall, x, 10), rand10 = at(rand, x, 10), norm10 = (recall10 - rand10) / (1 - rand10),
            norm_area = sum(recall - rand) / sum(1 - rand), .groups = "drop") %>%
  arrange(desc(norm5))
print(tab, n = Inf)
write_csv(tab, file.path(work_dir, cfg$results, "benchmark_parse_summary.csv"))
write_csv(curves %>% select(series, k = x, recall, rand), file.path(work_dir, cfg$results, "benchmark_parse_recall_curve.csv"))
