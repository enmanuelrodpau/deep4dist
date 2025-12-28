
## ---- Convert raw point clouds (.laz) to DSM files (.tif) ---####
# 1- Reads in the raw point clouds, extract the height variable, and convert it to a raster 
# 2- Save it to a file
# 3- Remove the raw point cloud file


library(lidR)
library(raster)

library(doSNOW)
library(foreach)

# ----------------------------
# Settings / paths
# ----------------------------

pth <- "./dataset" # Path to directory where the data is/will be stored
  
bdom_dir <- "dop_download/bdom" # Directory where the raw point clouds where downloaded 

out_dir <- "dsm" # Directory where the DSM will be saved

# Create out_dir if it does not exists
dir.create(file.path(pth, out_dir), recursive = T, showWarnings = F)

# Since this is a very memory-intensive preprocessing files that has been preprocessed are first filtered out 
out_files <- list.files(file.path(pth, out_dir), pattern = ".tif$")

bdom_files <- list.files(file.path(pth, bdom_dir), full.names=T, pattern = ".laz$")

if(length(out_files) > 0 ){
	out_files <- gsub(".tif", ".laz", out_files)
	bdom_files <- bdom_files[which(!basename(bdom_files) %in% out_files)]
}

# ----------------------------
# Parallel setup + progress bar
# ----------------------------
cl <- parallel::makeCluster(9) # Select an appropiate number of cores depending on the available RAM
doSNOW::registerDoSNOW(cl)

iterations <- length(bdom_files)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


# ----------------------------
# Convert point cloud to DSM
# ----------------------------

foreach(i = 1:iterations, .packages = c( "raster", "lidR"), .options.snow = opts, .verbose = T)%dopar%{
  xyz <- readLAS(bdom_files[i], select = "xyz")
  ref_crs <- crs(xyz)
  xyz <- data.frame(x=xyz$X, y = xyz$Y, z = xyz$Z)
  
  try({
    
    xyz0 <- SpatialPixelsDataFrame(xyz[,c('x','y')], data.frame(z = xyz[,c('z')]), proj4string = ref_crs,
                                   tolerance = 0.916421)
    xyz0 <- raster(xyz0[,'z'])
    
    bname <- gsub(".laz",".tif",basename(bdom_files[i]))
    
    writeRaster(xyz0,
                file.path(pth, paste(out_dir, bname, sep="/")),
                format ="GTiff", 
                overwrite=F)
    file.remove(bdom_files[i])
    rm(xyz)
    rm(xyz0)
  })
}

stopCluster(cl)
rm(cl)
