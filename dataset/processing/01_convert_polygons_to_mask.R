
## ---- Convert disturbance vector layer to masks ---####
# 1- Reads the DOP grid and select those that were downloaded from the url file (dop_download/rgbi_download_url.txt)
# 2- Reads the disturbance polygons, crop the polygons to the extent of the DOP frames, and filter disturbances based on a given MMU
# 3- Convert the disturbance plolygons to raster (.tif)
 
library(doSNOW)
library(foreach)

library(raster)
library(sf)

library(dplyr)
library(purrr)

# ----------------------------
# Helper(s)
# ----------------------------

crop_polygon_to_bbox <- function(data,bb){
  return(sf::st_crop(data, bb))
}

# ----------------------------
# Settings / paths
# ----------------------------

pth <- "./dataset" # Path to directory where the data is/will be stored

dop_dir <- "dop_download/dop_image" # Directory where the DOP images were downloaded

out_dir <- file.path(pth, "dop_mask") # Directory where the rasterized masks will be saved

dir.create(out_dir, showWarnings = F)

mmu <- 100 # Minimum mapping unit (area threshold)

### Input files
dist_data <- st_read(file.path(pth, "vector/rlp_disturbance_layer_v1_2.gpkg")) 

rlp_grid <- read_sf(file.path(pth, "aux/dop_tile_grid_rp.gpkg")) 

selected_rlp_dop_frames <- read.csv(file.path(pth,"/dop_download/rgbi_download_url.txt"), header = F)[,1]

selected_rlp_dop_frames <- gsub(".jp2","",basename(selected_rlp_dop_frames))

rlp_grid_sub <- rlp_grid[unlist(st_intersects(dist_data,rlp_grid)),]%>% 
  distinct()

# Filter disturbances intersecting the downloaded DOP frames
selected_rlp_dop_frame_ids <- substr(selected_rlp_dop_frames, 11,21)

rlp_grid_sub <- rlp_grid_sub %>% 
  mutate(target_dop_id <- paste(substr(new_name, 10,11), 
                               substr(new_name, 12,19), sep ="_")
         ) %>% 
  filter(target_dop_id %in% selected_rlp_dop_frame_ids)

# ----------------------------
# Parallel setup + progress bar
# ----------------------------

cl <- parallel::makeCluster(parallelly::availableCores(omit = 1))

doSNOW::registerDoSNOW(cl)

iterations <- nrow(rlp_grid_sub)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

# ----------------------------
# 1) Crop disturbances to each DOP bbox
# ----------------------------

cropped_data <- foreach(i = 1:iterations, .packages = c("sf", "dplyr"), .options.snow = opts, .verbose = T)%dopar%{
  
  tryCatch({crop_polygon_to_bbox(dist_data,rlp_grid_sub[i,]) %>% 
      mutate(area = as.numeric(st_area(geom)))},
      error=function(e){e$message})
  
}

names(cropped_data) <- rlp_grid_sub$target_dop_id

# filter only disturbance polygons within bbox and greather than mmu

errors <- cropped_data[unlist(cropped_data %>% map(~nrow(.x)) < 1) ]

ft_dop_id <- rlp_grid_sub %>% 
  filter(!target_dop_id %in% names(errors)) %>% 
  st_drop_geometry() %>% 
  split(.,seq(nrow(.)))

cropped_data_filtered <- bind_rows (cropped_data[unlist(cropped_data %>% map(~nrow(.x)) > 0) ] %>% 
                                     map2(.,ft_dop_id, ~{.x %>% mutate(target_dop_id = .y)})) %>% 
  filter(area >= mmu)%>% 
  distinct()


# ----------------------------
# 2) Rasterize per DOP tile
# ----------------------------

foreach(i = 1:iterations, .packages = c( "raster", "dplyr", "sf"), .options.snow = opts, .verbose = T)%dopar%{
  
  
  try({
    
    grid_sample <- rlp_grid_sub[i,]
    
    target_dop_id <- grid_sample$target_dop_id
    
    
    ex <- st_crop(cropped_data_filtered, grid_sample)%>% 
      st_collection_extract(
        .,
        type = c("POLYGON", "POINT", "LINESTRING"),
        warn = FALSE
      )
    
    ex <- ex %>%
      mutate(Dist = as.numeric(Dist), area = as.numeric(st_area(geom))) %>%
      filter(area >= mmu) %>% 
      st_cast("MULTIPOLYGON")%>% 
      st_cast("POLYGON")
    
    
    bname <- grep(target_dop_id, selected_rlp_dop_frames, value = TRUE, fixed = TRUE)
    
    template <- raster(file.path(pth, dop_dir, paste0(bname, ".jp2")))
    
    vect_rast <- raster::rasterize((ex), template, field= "Dist", background=0)
    
    crs(vect_rast) <- crs(ex)
    
    # 
    file_name <- file.path(out_dir, bname, "_", substr(unique(grid_sample$erstellung), 7,11) ,".tif")
    
    
    writeRaster(vect_rast,
                file_name,
                format="GTiff",
                datatype="INT1U",
                overwrite = T
    )
    
  }) 
  
}

# ----------------------------
# Cleanup
# ----------------------------

close(pb)

stopCluster(
  cl
) 

rm(cl)