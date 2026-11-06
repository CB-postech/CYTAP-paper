# CYTAP cytokine -> SecAct / CytoSig signature names, used to restrict and score SecAct and CytoSig
# to the CYTAP cytokines ("mapped" pool). Input: cytokine_ligand_map.csv (Cytokines, ligand_symbols).

# SecAct signatures are gene symbols: exact (case-insensitive) match of each listed ligand; IL-8 -> CXCL8.
secact_map <- function(map_raw, secact_ligands) {
  lookup <- setNames(secact_ligands, toupper(secact_ligands))
  map_raw$SecAct <- vapply(map_raw$ligand_symbols, function(s) {
    hit <- lookup[toupper(trimws(strsplit(s, "[,;/|]")[[1]]))]
    hit <- hit[!is.na(hit)]
    if (length(hit) == 0) NA_character_ else paste(unique(hit), collapse = ";")
  }, character(1), USE.NAMES = FALSE)
  map_raw$SecAct[map_raw$Cytokines == "IL-8"] <- "CXCL8"
  message(sprintf("SecAct: %d/%d cytokines mapped; unmapped: %s", sum(!is.na(map_raw$SecAct)), nrow(map_raw),
                  paste(map_raw$Cytokines[is.na(map_raw$SecAct)], collapse = ", ")))
  map_raw
}

# CytoSig signature labels are not always gene symbols (GCSF = CSF3, ...): curated label -> symbols.
CYTOSIG_TO_SYMBOL <- list(
  "Activin A" = "INHBA", "BDNF" = "BDNF", "BMP2" = "BMP2", "BMP4" = "BMP4", "BMP6" = "BMP6",
  "CD40L" = "CD40LG", "CXCL12" = c("CXCL12", "SDF1"), "EGF" = "EGF", "FGF2" = c("FGF2", "BFGF"),
  "GCSF" = "CSF3", "GDF11" = "GDF11", "GMCSF" = "CSF2", "HGF" = "HGF",
  "IFN1" = c("IFNB1", "IFNA1", "IFNA2", "IFNA"),                          # type I IFN
  "IFNG" = "IFNG",
  "IFNL" = c("IFNL1", "IFNL2", "IFNL3", "IL29", "IL28A", "IL28B"),         # type III IFN
  "IL10" = "IL10", "IL12" = c("IL12A", "IL12B"), "IL13" = "IL13", "IL15" = "IL15", "IL17A" = "IL17A",
  "IL1A" = "IL1A", "IL1B" = "IL1B", "IL2" = "IL2", "IL21" = "IL21", "IL22" = "IL22", "IL27" = "IL27",
  "IL3" = "IL3", "IL36" = c("IL36A", "IL36B", "IL36G"), "IL4" = "IL4", "IL6" = "IL6", "LIF" = "LIF",
  "LTA" = c("LTA", "TNFB"), "MCSF" = "CSF1", "NO" = NA_character_,           # nitric oxide: no gene
  "OSM" = "OSM", "TGFB1" = "TGFB1", "TGFB3" = "TGFB3", "TNFA" = "TNF", "TRAIL" = c("TNFSF10", "APO2L"),
  "TWEAK" = "TNFSF12", "VEGFA" = c("VEGFA", "VEGF"), "WNT3A" = "WNT3A"
)

cytosig_map <- function(map_raw, cytosig_ligands) {
  sym2cs <- do.call(c, lapply(names(CYTOSIG_TO_SYMBOL), function(cs) {
    syms <- toupper(c(cs, CYTOSIG_TO_SYMBOL[[cs]]))
    syms <- syms[!is.na(syms)]
    setNames(rep(cs, length(syms)), syms)
  }))
  sym2cs <- sym2cs[!duplicated(names(sym2cs))]
  stopifnot("CytoSig labels without a curated symbol" = all(cytosig_ligands %in% sym2cs))
  map_raw$CytoSig <- vapply(map_raw$ligand_symbols, function(s) {
    hit <- sym2cs[toupper(trimws(strsplit(s, "[,;/|]")[[1]]))]
    hit <- hit[!is.na(hit) & hit %in% cytosig_ligands]
    if (length(hit) == 0) NA_character_ else paste(unique(hit), collapse = ";")
  }, character(1), USE.NAMES = FALSE)
  # IL-23 and IL-35 share a subunit with IL-12 but are different cytokines: not mapped to the IL12 signature
  map_raw$CytoSig[map_raw$Cytokines %in% c("IL-23", "IL-35")] <- NA_character_
  message(sprintf("CytoSig: %d/%d cytokines mapped", sum(!is.na(map_raw$CytoSig)), nrow(map_raw)))
  map_raw
}

# rank a single condition: z-scores of the mapped signatures only; true_rank = best rank of the cytokine's signatures
score_mapped <- function(zmat, ct, cyt, cyt_to_lig, mapped, top_n = 10) {
  col <- paste(ct, cyt, sep = "|")
  if (!(col %in% colnames(zmat)))
    return(tibble(celltype = ct, cytokine = cyt, true_rank = NA_integer_, hit = FALSE, status = "no_result"))
  if (!(cyt %in% names(cyt_to_lig)) || is.na(cyt_to_lig[[cyt]]) || cyt_to_lig[[cyt]] == "")
    return(tibble(celltype = ct, cytokine = cyt, true_rank = NA_integer_, hit = FALSE, status = "no_ligand_map"))
  z <- zmat[, col]
  z <- z[names(z) %in% mapped]
  rt <- tibble(sig = names(z), z = z) %>% arrange(desc(z)) %>% mutate(rank = row_number())
  ranks <- rt$rank[rt$sig %in% strsplit(cyt_to_lig[[cyt]], ";")[[1]]]
  best <- if (length(ranks)) min(ranks) else Inf
  tibble(celltype = ct, cytokine = cyt, true_rank = ifelse(is.finite(best), best, NA_integer_),
         hit = is.finite(best) && best <= top_n, status = if (is.finite(best)) "scored" else "no_result")
}

# signature x cell type activity -> ranking table for one cell type: rank over all signatures, mapped_rank over
# the signatures mapped to CYTAP cytokines (NA for the others), CYTAP = mapped cytokine(s)
rank_activity <- function(act, cell_type, map_col, map) {
  df <- data.frame(signature = rownames(act$zscore), beta = act$beta[, cell_type], se = act$se[, cell_type],
                   zscore = act$zscore[, cell_type], pvalue = act$pvalue[, cell_type])
  df$FDR <- p.adjust(df$pvalue, method = "BH")
  mp <- map[!is.na(map[[map_col]]) & map[[map_col]] != "", ]
  if (map_col == "CytoSig") {   # one signature can stand for several cytokines (IFN1, IFNL, ...): "a/b"
    long <- tibble(signature = strsplit(mp[[map_col]], ";"), Cytokines = mp$Cytokines) %>% unnest(signature)
    lookup <- tapply(long$Cytokines, long$signature, function(x) paste(unique(x), collapse = "/"))
  } else {                      # SecAct, as in the original analysis: whole map entry, first cytokine
    if (any(grepl(";", mp[[map_col]])))
      message("SecAct entries with several signatures (not split here): ",
              paste(mp$Cytokines[grepl(";", mp[[map_col]])], collapse = ", "))
    lookup <- setNames(mp$Cytokines, mp[[map_col]])
  }
  df$CYTAP <- as.character(lookup[df$signature])   # plain vector (tapply gives a 1-d array)
  df$rank <- rank(-df$zscore, ties.method = "min")
  mapped <- !is.na(df$CYTAP)        # rank among the signatures mapped to CYTAP cytokines
  df$mapped_rank <- NA_integer_
  df$mapped_rank[mapped] <- as.integer(rank(-df$zscore[mapped], ties.method = "min"))
  df %>% arrange(rank) %>% select(rank, mapped_rank, signature, CYTAP, zscore, beta, se, pvalue, FDR)
}
