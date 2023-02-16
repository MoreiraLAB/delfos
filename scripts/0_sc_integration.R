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

# KAGOHARA_SCC1
kag <- read.csv("Kagohara/SCC1Matrix.csv")
kag <- kag[kag[,1] %in% anno[, 1], ]
kag <- kag[!duplicated(kag[, 1]),]
rownames(kag) <- kag[, 1]
kag <- kag[ , grepl( "PBS" , colnames(kag))]

kagohara1 <- CreateSeuratObject(counts = kag, min.cells = 0, assay = "RNA", project = "Kagohara", min.features = 100)

# KAGOHARA_SCC25
kag <- read.csv("Kagohara/SCC25Matrix.csv")
kag <- kag[kag[,1] %in% anno[, 1], ]
kag <- kag[!duplicated(kag[, 1]),]
rownames(kag) <- kag[, 1]
kag <- kag[ , grepl( "PBS" , colnames(kag))]

kagohara2 <- CreateSeuratObject(counts = kag, min.cells = 0, assay = "RNA", project = "Kagohara", min.features = 100)

# BENDAVID
ben <- read.csv("BenDavid/GSE114461.csv")
ben <- ben[rownames(ben) %in% anno[, 1], ]
ben <- ben[ , grepl( "t0" , colnames(ben))]

bendavid <- CreateSeuratObject(counts = ben, project = "BenDavid" , min.cells = 0, assay = "RNA", min.features = 100)

# MCFARLAND
mcfarland <- Read10X("McFarland/")
mcfarland <- mcfarland[rownames(mcfarland) %in% anno[, 1], ]
mcfarland <- CreateSeuratObject(counts = mcfarland, min.cells = 0, assay = "RNA", project = "mcfar", min.features = 100)

# SCHNEPP
schnepp <- read.csv("Schnepp/GSE140440.csv", header = T)
schnepp <- separate(schnepp, col = 'X0', into = c('bla', 'genes'), sep = '\\|')
schnepp <- schnepp[, -1]
schnepp <- schnepp[!duplicated(schnepp[, 'genes']),]
rownames(schnepp) <- schnepp[, 1]
anno <- read.csv("hGENES.csv", header = T)
schnepp <- schnepp[, -1]
schnepp <- schnepp[rownames(schnepp) %in% anno[, 1], ]

schnepp <- CreateSeuratObject(counts = schnepp, min.cells = 0, assay = "RNA", project = "Schnepp", min.features = 100)

# SRIRAMKUMAR
srilanka <- Read10X("Sriramkumar/")
srilanka <- srilanka[rownames(srilanka) %in% anno[, 1], ]
colnames(srilanka) <- paste0(colnames(srilanka), "_", "OVCAR3")

sriramkumar <- CreateSeuratObject(counts = srilanka, min.cells = 0, assay = "RNA", project = "Sriramkumar", min.features = 100)

# Integrate the datasets
data_ais <- c(kagohara1, kagohara2, bendavid, mcfarland, schnepp, sriramkumar)
data_ais <- lapply(X = data_ais, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)})

features <- SelectIntegrationFeatures(object.list = data_ais)
anchors <- FindIntegrationAnchors(object.list = data_ais, anchor.features = features)
data_s <- IntegrateData(anchorset = anchors)

data_s <- subset(data_s, subset = nFeature_RNA > 100 & nFeature_RNA < 8000)
s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes
data_s <- CellCycleScoring(data_s, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
data_s <- ScaleData(data_s, vars.to.regress = c("S.Score", "G2M.Score"))

datas <- as.data.frame(t(data_s@assays$integrated@scale.data))

write.csv(datas, paste(DEFAULT_LOCATION, "data/single-cell/integrated_sc.csv", sep = ""), row.names = T)

# If you with to save the RDS file of the integrated Seurat object, untoggle the comment below:
# saveRDS(data_s, file = paste(DEFAULT_LOCATION, "data/single-cell/sc_integration/integrated_sc.rds", sep = ""))











