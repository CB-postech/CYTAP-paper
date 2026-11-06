# Query datasets: QC, normalization and annotation (Seurat). Run from the repository root:
#   Rscript 00_preprocessing/preprocess.R {ifnb | macrophage | gcsf} [RAW_DIR]
# RAW_DIR (default data/raw) holds the downloaded data: IFNb_data/, macrophage_stimulation/, G-CSF_BM/.
# Log-normalized h5ad files for the State embedding are written to data/normalized/ (00_preprocessing/state_embeddings.sh).
#   ifnb       IFN-beta-stimulated PBMCs (Kang et al. 2018; GEO GSE96583, batch 2): ctrl + stim 10x matrices,
#              singlets, LogNormalize -> data/normalized/IFNb_data_total.h5ad
#   macrophage cytokine-stimulated macrophages (Zhang et al. 2021; GEO GSE168710): raw counts of the four samples
#              placed into the published Seurat object (cells and metadata of the original study), exported as
#              counts.mtx / metadata / gene names (h5ad: preprocess.py macrophage)
#   gcsf       G-CSF-primed bone marrow (You et al. 2022; GEO GSE193138; 2 donors, pre / post): QC (>= 1,000 UMI,
#              >= 300 genes, < 10% mitochondrial), LogNormalize, PCA (30), Harmony (donor), clustering (res. 0.8),
#              lineage annotation -> data.gcsf_seurat, data.gcsf_counts (raw counts in .raw.X)
# IL-7 T cells (Jang et al. 2026): non-T cells removed and CD4_T / CD8_T annotated per cohort (CRC, GBM); see Methods.
source("R/paths.R")
suppressPackageStartupMessages({ library(Seurat); library(SeuratDisk); library(harmony); library(Matrix); library(dplyr) })
args <- commandArgs(trailingOnly = TRUE)
dataset <- match.arg(args[1], c("ifnb", "macrophage", "gcsf"))
RAW <- if (length(args) > 1) args[2] else file.path(work_dir, "data/raw")
NORM <- file.path(work_dir, "data/normalized")
dir.create(NORM, recursive = TRUE, showWarnings = FALSE)

save_h5ad <- function(obj, h5seurat) {   # h5ad next to the .h5Seurat file
  dir.create(dirname(h5seurat), recursive = TRUE, showWarnings = FALSE)
  SaveH5Seurat(obj, filename = h5seurat, overwrite = TRUE)
  Convert(h5seurat, dest = "h5ad", overwrite = TRUE)
}

# ---------------------------------------------------------------- IFN-beta PBMC
if (dataset == "ifnb") {
  d <- file.path(RAW, "IFNb_data")
  tsne <- read.table(file.path(d, "GSE96583_batch2.total.tsne.df.tsv"), row.names = 1, header = TRUE, sep = "\t")
  total <- CreateSeuratObject(cbind(Read10X(file.path(d, "Ctrl")), Read10X(file.path(d, "treatment"))))
  total@meta.data <- cbind(total@meta.data, tsne[colnames(total), ])     # stim, ind, cell, multiplets
  total <- NormalizeData(total)
  total <- total[, total$multiplets == "singlet"]
  save_h5ad(total, file.path(NORM, "IFNb_data_total.h5Seurat"))
}

# ---------------------------------------------------------------- macrophages
if (dataset == "macrophage") {
  d <- file.path(RAW, "macrophage_stimulation")
  macrophage_stim <- readRDS(file.path(d, "macrophage_stim.rds"))   # published Seurat object
  files <- c(sample_1 = "GSM5159500_BRI-783", sample_2 = "GSM5159501_BRI-785",
             sample_3 = "GSM5159502_BRI_787", sample_4 = "GSM5159503_BRI_789")
  mats <- lapply(names(files), function(s) {
    m <- Read10X_h5(file.path(d, "GSE168710_RAW", paste0(files[[s]], "_filtered_feature_bc_matrix.h5")))[["Gene Expression"]]
    colnames(m) <- paste0(s, colnames(m))
    m
  })
  stopifnot(all(sapply(mats, function(m) identical(rownames(m), rownames(mats[[1]])))))
  counts <- do.call(cbind, mats)
  counts <- counts[, colnames(counts) %in% colnames(macrophage_stim)]
  stopifnot(all(colnames(counts) == colnames(macrophage_stim)))
  out <- file.path(d, "data")
  dir.create(out, showWarnings = FALSE)
  write.csv(macrophage_stim@meta.data[, c("sample", "stim", "barcode")], file.path(out, "macrophage_stim_metadata.csv"),
            quote = FALSE, row.names = FALSE)
  writeMM(counts, file.path(out, "macrophage_stim_counts.mtx"))
  write.table(data.frame(gene = rownames(counts)), file.path(out, "macrophage_stim_gene_names.csv"),
              quote = FALSE, row.names = FALSE, col.names = FALSE)
}

# ---------------------------------------------------------------- G-CSF bone marrow
if (dataset == "gcsf") {
  d <- file.path(RAW, "G-CSF_BM", "GSE193138_RAW")
  samples <- data.frame(donor = c("Donor1", "Donor2", "Donor1", "Donor2"), condition = c("pre", "pre", "post", "post"),
                        prefix = c("GSM5774293_Donor1_pre", "GSM5774294_Donor2_pre",
                                   "GSM5774295_Donor1_post", "GSM5774296_Donor2_post"))
  objs <- lapply(seq_len(nrow(samples)), function(i) {
    s <- samples[i, ]
    mat <- ReadMtx(mtx = file.path(d, paste0(s$prefix, "_matrix.mtx.gz")),
                   cells = file.path(d, paste0(s$prefix, "_barcodes.tsv.gz")),
                   features = file.path(d, paste0(s$prefix, "_features.tsv.gz")), feature.column = 2)
    o <- CreateSeuratObject(counts = mat, project = s$prefix, min.cells = 3, min.features = 200)
    o$donor <- s$donor; o$condition <- s$condition; o$sample <- s$prefix
    o
  })
  bm <- merge(objs[[1]], y = objs[-1], add.cell.ids = samples$prefix, project = "G_BM")
  bm[["percent.mt"]] <- PercentageFeatureSet(bm, pattern = "^MT-")
  bm <- subset(bm, subset = nCount_RNA >= 1000 & nFeature_RNA >= 300 & percent.mt < 10)
  bm <- NormalizeData(bm) %>% FindVariableFeatures(nfeatures = 2000) %>% ScaleData() %>% RunPCA(npcs = 30)
  emb <- HarmonyMatrix(data_mat = Embeddings(bm, "pca"), meta_data = bm@meta.data, vars_use = "donor", do_pca = FALSE)
  bm[["harmony"]] <- CreateDimReducObject(embeddings = emb, key = "harmony_", assay = DefaultAssay(bm))
  bm <- RunUMAP(bm, reduction = "harmony", dims = 1:30) %>% FindNeighbors(reduction = "harmony", dims = 1:30) %>%
    FindClusters(resolution = 0.8)
  lineage <- list(T_NK = c(0, 1, 2, 4, 5, 19), HSPC = c(13, 14), EM_lineage = c(8, 9, 10),
                  B_lineage = c(11, 12, 16, 18, 21), Mye_lineage = c(3, 6, 7, 15, 17, 20, 22))
  bm$celltype <- "NA"
  for (l in names(lineage)) bm$celltype[bm$seurat_clusters %in% lineage[[l]]] <- l
  print(table(bm$celltype, bm$condition))
  dir.create(dirname(cfg_path("data", "gcsf_seurat")), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bm, cfg_path("data", "gcsf_seurat"))
  counts_obj <- CreateSeuratObject(counts = GetAssayData(bm, assay = "RNA", slot = "counts"), meta.data = bm@meta.data)
  save_h5ad(counts_obj, sub("\\.h5ad$", ".h5Seurat", cfg_path("data", "gcsf_counts")))   # raw counts in .raw.X
}
