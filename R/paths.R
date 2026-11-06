# Paths from config/paths.yaml (needs only the yaml package). Run scripts from the repository root.
cfg <- suppressWarnings(yaml::read_yaml("config/paths.yaml"))
if (file.exists("config/paths.local.yaml"))   # machine-specific entries (not tracked), e.g. tool folders
  cfg <- utils::modifyList(cfg, suppressWarnings(yaml::read_yaml("config/paths.local.yaml")))
wd <- cfg$work_dir
if (is.null(wd) || is.na(wd) || !is.character(wd)) wd <- "."   # unquoted `work_dir: .` is read as a number (NA)
work_dir <- Sys.getenv("CYTAP_WORK_DIR", wd)
if (!grepl("^(/|[A-Za-z]:)", work_dir)) work_dir <- normalizePath(work_dir)   # relative to the repository root

# cfg_path("data", "gcsf_seurat") -> absolute path (relative entries resolved against work_dir)
cfg_path <- function(...) {
  p <- cfg
  for (k in c(...)) p <- p[[k]]
  if (is.null(p)) stop("config entry not found: ", paste(c(...), collapse = "."))
  if (grepl("^(/|[A-Za-z]:)", p)) p else file.path(work_dir, p)
}

# file under benchmark.parse_dir / benchmark.in_vivo_dir (folders created)
bench_path <- function(kind = c("parse_dir", "in_vivo_dir"), ...) {
  p <- file.path(cfg_path("benchmark", match.arg(kind)), ...)
  dir.create(dirname(p), recursive = TRUE, showWarnings = FALSE)
  p
}

# refuse to overwrite an existing output unless FORCE=1
new_file <- function(p) {
  if (file.exists(p) && Sys.getenv("FORCE") != "1") stop(p, " exists (set FORCE=1 to replace)")
  p
}

# NicheNet v2 prior (Zenodo 7074291): read from reference.nichenet_dir, downloaded there on first use
nichenet_prior <- function(file) {
  p <- file.path(cfg_path("reference", "nichenet_dir"), file)
  if (!file.exists(p)) {
    dir.create(dirname(p), recursive = TRUE, showWarnings = FALSE)
    options(timeout = max(3600, getOption("timeout")))
    download.file(paste0("https://zenodo.org/records/7074291/files/", file), paste0(p, ".part"), mode = "wb")
    file.rename(paste0(p, ".part"), p)
  }
  readRDS(p)
}
