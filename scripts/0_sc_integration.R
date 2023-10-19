library(dplyr)
library(Seurat)
library(patchwork)
library(Matrix)
library(tidyr)
library(reticulate)

reticulate::source_python('delfos_resources.py')
setwd(paste(DEFAULT_LOCATION, "data/single-cell/sc_integration/", sep = ""))

"
Each dataset is first called and processed into a Seurat object. The resulting objects are then integrated and processed according to Seurat's
introductory vignette.

"

# anno represents the genes in common between all datasets
anno <- read.csv("hGENES.csv", header = T)

mt_cutoff <- 10
lower_cutoff <- 1000

# KAGOHARA_SCC25
kag <- read.csv("Kagohara/SCC25Matrix.csv")
kag <- kag[kag[,1] %in% anno[, 1], ]
kag <- kag[!duplicated(kag[, 1]),]
rownames(kag) <- kag[, 1]
kag <- kag[ , grepl( "PBS" , colnames(kag))]

kagohara2 <- CreateSeuratObject(counts = kag, min.cells = 0, assay = "RNA", project = "Kagohara", min.features = lower_cutoff)
kagohara2[["percent.mt"]] <- PercentageFeatureSet(kagohara2, pattern = "^MT-")
kagohara2 <- subset(kagohara2, percent.mt < mt_cutoff)

# BENDAVID
ben <- read.csv("BenDavid/GSE114461.csv")
ben <- ben[rownames(ben) %in% anno[, 1], ]
ben <- ben[ , grepl( "t0" , colnames(ben))]

bendavid <- CreateSeuratObject(counts = ben, project = "BenDavid" , min.cells = 0, assay = "RNA", min.features = lower_cutoff)
bendavid[["percent.mt"]] <- PercentageFeatureSet(bendavid, pattern = "^MT-")
bendavid <- subset(bendavid, percent.mt < mt_cutoff)

# MCFARLAND
mcfarland <- Read10X("McFarland/")
mcfarland <- mcfarland[rownames(mcfarland) %in% anno[, 1], ]

mcfarland <- CreateSeuratObject(counts = mcfarland, min.cells = 0, assay = "RNA", project = "mcfar", min.features = lower_cutoff)
mcfarland[["percent.mt"]] <- PercentageFeatureSet(mcfarland, pattern = "^MT-")
mcfarland <- subset(mcfarland, percent.mt < mt_cutoff)

# SCHNEPP
schnepp <- read.csv("Schnepp/GSE140440.csv", header = T)
schnepp <- separate(schnepp, X0, c(NA, 'genes'), sep = "\\|")
schnepp <- schnepp[!duplicated(schnepp[, 'genes']),]
rownames(schnepp) <- schnepp[, 1]
schnepp <- schnepp[, -1]
schnepp <- schnepp[rownames(schnepp) %in% anno[, 1], ]

schnepp <- CreateSeuratObject(counts = schnepp, min.cells = 0, assay = "RNA", project = "Schnepp", min.features = lower_cutoff)
schnepp[["percent.mt"]] <- PercentageFeatureSet(schnepp, pattern = "^MT-")
schnepp <- subset(schnepp, percent.mt < mt_cutoff)

# SRIRAMKUMAR
srilanka <- Read10X("Sriramkumar/")
srilanka <- srilanka[rownames(srilanka) %in% anno[, 1], ]
colnames(srilanka) <- paste0(colnames(srilanka), "_", "OVCAR3")

sriramkumar <- CreateSeuratObject(counts = srilanka, min.cells = 0, assay = "RNA", project = "Sriramkumar", min.features = lower_cutoff)
sriramkumar[["percent.mt"]] <- PercentageFeatureSet(sriramkumar, pattern = "^MT-")
sriramkumar <- subset(sriramkumar, percent.mt < mt_cutoff)

merged_dataset <- merge(kagohara2, y = c(bendavid, mcfarland, schnepp, sriramkumar),
                        add.cell.ids = c("kagohara2", "ben-david", "mcfarland", "schnepp", "sriramkumar"))

s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes

data_s <- subset(merged_dataset, subset = nFeature_RNA > 1000 & nFeature_RNA < 7500)
data_s <- CellCycleScoring(data_s, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
data_s <- NormalizeData(data_s)
data_s <- FindVariableFeatures(data_s, selection.method = "vst", nfeatures = 2000)
data_s <- ScaleData(data_s)

datas <- as.data.frame(t(data_s@assays$RNA@scale.data))

write.csv(datas, paste(DEFAULT_LOCATION, "data/single-cell/integrated_sc.csv", sep = ""), row.names = T)

# If you with to save the RDS file of the integrated Seurat object, untoggle the comment below:
# saveRDS(data_s, file = paste(DEFAULT_LOCATION, "data/single-cell/sc_integration/integrated_sc.rds", sep = ""))











