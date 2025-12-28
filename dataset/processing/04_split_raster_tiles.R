## ---- Create equal-sized tiles ---####
# 1- Reads the disturbance masks, split in tiles (500x500 pixels) and save only where disturbances exists
# 2- Crop DOP images to masks tiles

require(raster)
require(sf)

require(dplyr)
require(curl)

require(doSNOW)
require(purrr)

# ----------------------------
# Helper(s)
# ----------------------------

splitRaster <- function(x=raster, tile_size=c(256, 256), out_dir="", base_name ="", mmu = 1000){
  
  y_n_tile <- nrow(x)/ tile_size[1] 
  
  x_n_tile <- ncol(x)/ tile_size[2]
  
  rows <- seq(0, nrow(x), by = tile_size[1])
  rows <- apply(embed(rows, 2), 1, rev)
  rows[1, ] <- rows[1,]+1
  
  cols <- seq(0, ncol(x), by = tile_size[2])
  cols <- apply(embed(cols, 2), 1, rev)
  cols[1, ] <- cols[1,]+1
  
  tile_rows <- lapply(1:ncol(rows), function(n){
    
    crop(x, extent(x, rows[1,n], rows[2,n], 1, ncol(x)))
    
  })
  
  
  tile_cols <- lapply(tile_rows, function(tile_row){
    
    lapply(1:ncol(cols), function(n){
      crop(tile_row, extent(tile_row, 1, nrow(tile_row), cols[1,n], cols[2,n]))
    })
    
    
  })
  
  lapply(1:length(tile_cols), function(n) {
    tmp_split = tile_cols[[n]]
    
    row_seq <- sprintf("%03d", c(1:length(tile_cols)))
    
    lapply(1:length(tmp_split), function(z){
      
      col_seq <- sprintf("%03d", c(1:length(tmp_split)))
      
      r <- tmp_split[[z]]
      
      rr <- cellStats(r, max)
      
      
      if(rr > 0){
        writeRaster(r, filename = paste(out_dir, 
                                        paste0(base_name,"_", row_seq[n], "_",col_seq[z], ".tif"), 
                                        sep="/"),
                    format="GTiff",
                    datatype = "INT1U",
                    overwrite=F)
      }
      
      
      
      
    })
    
    
    
  })
  
  
}

# ----------------------------
# Settings / paths
# ----------------------------

split_ <- ""

pth <- "./dataset"

out_dir <- pth

dop_pth = "dop_download/dop_image"

ndsm_pth = "ndsm"

# Creates the forlder to save the images and masks
mask_dir_temp <- "mask_temp" # Directory to save all masks

mask_dir <- "mask" # Directory to save final masks after filtering

image_dir <- "image" # Directory to save the rgbie images

log_dir <- "logs"

temp_dir <- "stats"

c(mask_dir, 
  mask_dir_temp, 
  image_dir, 
  temp_dir, 
  log_dir) %>% map( ~dir.create( file.path(out_dir, split_, .x), 
                                recursive = T
                                )
                    )


tile_size <- c(500, 500)

mmu <- 100

rast_split <- list.files(file.path(pth,"dop_mask"), full.names = T, pattern = ".tif$")

# ----------------------------
# Parallel setup + progress bar
# ----------------------------
cl <- parallel::makeCluster(parallelly::availableCores(omit = 1))

doSNOW::registerDoSNOW(cl)

iterations <- length(rast_split)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

# ----------------------------
# 1) DOP masks tilling
# ----------------------------
foreach(i = 1:iterations, .packages = c( "raster", "dplyr"), .options.snow = opts, .verbose = T)%dopar%{

  x = raster(rast_split[i])

  base_name = gsub(".tif","",basename(rast_split[i]))

  try({

  splitRaster(x, tile_size=tile_size,
              out_dir=file.path(out_dir, split_, mask_dir_temp),
              base_name = base_name,
              mmu = mmu)
})
}


# ----------------------------
# 2) DOP images tilling
# ----------------------------

#Get all masks
masks = list.files(file.path(pth, mask_dir_temp), full.names = T, pattern = ".tif$")

bname_masks = basename(masks)

#Get the id of masks (original from dop images) and load and filter dop rgbi and ndsm
target_dop_id = unique(substr(bname_masks, 11, 21))

dop_images = list.files(dop_pth, pattern = ".jp2$", full.names = T)

dop_images = unlist(lapply(target_dop_id, function(x)   grep(x, dop_images, value = TRUE, fixed = TRUE)))

ndsm_images = list.files(ndsm_pth, pattern = ".tif$", full.names = T)

ndsm_images = unlist(lapply(target_dop_id, function(x)   grep(x, ndsm_images, value = TRUE, fixed = TRUE)))

# Since proj library is giving errors with the EPSG:25832, here I passed the crs definition

ref_crs =crs("+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs")

iterations <- length(masks)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

# ----------------------------
# Run the tilling
# ----------------------------

foreach(i = 1:iterations, .packages = c( "raster", "dplyr", "stringr"), .options.snow = opts, .verbose = T)%dopar%{
  
  msk <- masks[i]  
  
  filename <- paste(out_dir, 
                   log_dir, gsub(".tif", ".txt", basename(msk)), sep="/")
  
  if(!file.exists(filename)){
    
    write.table(filename, filename)
    
    
    c_mask <- raster(msk, crs = ref_crs)
    
    
    val <- getValues(c_mask)
    
    
    rr <- table(val)
    
    ress <- res(c_mask)[1]
    
    rr <- data.frame(dist = names(rr), area = as.numeric(rr) * ress**2)
    
    rr <- rr[rr[,1] >0, ] %>% 
      mutate(mask_name = substr(basename(msk), 1 , 39))
    
    if(any(rr[,2] > mmu)){ # Filter disturbance events > 100 m2
      
      if(any(rr[,2] < mmu)){ # if there are more than 1 event per tile and one of them is < 100 m2 this line makes it background
        
        ind <- which(rr[,2] < mmu) 
        c_mask[c_mask== ind] <- 0  
      }
      
      
      # Read, crop and stack
      
      dop_id <- unique(substr(basename(msk), 11, 21))
      
      c_dop <- grep( dop_id, dop_images, value = TRUE, fixed = TRUE)
      
      c_ndsm <- grep( dop_id, ndsm_images, value = TRUE, fixed = TRUE)
      
      c_dop <- stack(c_dop)
      
      crs(c_dop) <- ref_crs
      
      c_ndsm <- raster(c_ndsm, crs = ref_crs)
      
      c_min <- cellStats(c_ndsm, min)
      c_max <- cellStats(c_ndsm, max)
      
      c_ndsm <- round(255*(c_ndsm-c_min)/(c_max - c_min)) # normalize the ndsm to be 0 - 255 for DL input #original <- (scaled / 255) * (c_max - c_min) + c_min
      
      
      rgbie <- stack(c_dop, c_ndsm)
      
      c_rgbie <- crop(rgbie, c_mask)
      
      writeRaster(c_rgbie, filename = paste(out_dir, 
                                            image_dir, basename(msk), sep="/"),
                  format="GTiff",
                  datatype = "INT1U",
                  overwrite=T)
      
      writeRaster(c_mask, filename = paste(out_dir, 
                                           mask_dir, basename(msk), sep="/"),
                  format="GTiff",
                  datatype = "INT1U",
                  overwrite=T)
      
      
      
    }
  }
}

# ----------------------------
# Cleanup
# ----------------------------
close(pb)
stopCluster(cl)
rm(cl)
