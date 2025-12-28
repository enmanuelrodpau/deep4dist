
## ---- Create nDSM files from preprocessed point clouds and DEM ---####
# 1- Reads in the DEM mosaic (vrt..) and individual (DSM-preprocessed point clouds)
# 2- Crop the DEM to the extent of DSM and resample  it to the DSM spatial resolution
# 3- Calculate the nDSM (object height) and set minimum to height 0
# 4- Save nDSM to out_dir

library(raster)

library(dplyr)

library(doSNOW)
library(foreach)

# ----------------------------
# Helper(s)
# ----------------------------

create_vrt <- function(dem_dir_path){
  pass_command <- paste0("gdalbuildvrt ", dem_dir_path,"/dem_rlp.vrt ", dem_dir_path,"/*.tif")
  print("Starting DEM vrt creation.....")
  system(pass_command, intern = T)
  
}

# ----------------------------
# Settings / paths
# ----------------------------

pth <- "./dataset" # Path to dir where the raw data was downloaded

dem_dir <- "dop_download/dem" #DEM 

bdom_dir <- "dsm" # Preprocessed point clouds

out_dir <- "ndsm" # Output directory

dir.create(file.path(pth, out_dir), recursive = T, showWarnings = F)

create_vrt(file.path(pth, dem_dir)) # Creates the DEM vrt file

dem_list <- list.files(file.path(pth, dem_dir), full.names = T, pattern = ".vrt$") # Assumes you have created a virtual raster..

bdom_list <- list.files(file.path(pth, bdom_dir), full.names = T, pattern = ".tif$")

# ----------------------------
# Parallel setup + progress bar
# ----------------------------

cl <- parallel::makeCluster(parallelly::availableCores(omit = 1))

doSNOW::registerDoSNOW(cl)

iterations <- length(bdom_list)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


# ----------------------------
# Create nDSM point cloud to DSM
# ----------------------------
foreach(i = 1:iterations, .packages = c( "raster",  "dplyr"), .options.snow = opts, .verbose = T)%dopar%{
  
  c_file <- bdom_list[i]
  
  file_out <- paste(pth, out_dir, basename(c_file), sep="/")
  
  file.exists(!file_out){      
    
    
    dem <- raster(dem_list)
    bdom <- raster(c_file)
    
    c_dem <- crop(dem, bdom)
    
    c_dem <- raster::resample(c_dem, bdom,method= "bilinear")
    
    ndsm <- bdom - c_dem
    
    ndsm[ndsm < 0] <- 0
    
    writeRaster(ndsm, file_out,
                format = "GTiff",
                overwrite=T)
  }
}

# ----------------------------
# Cleanup
# ----------------------------

close(pb)

stopCluster(
  cl
) 

rm(cl)