# Dataset Creation
This repository hosts the code used to create the dataset in the paper.

## Structure
- *dop_download*: links to download the raw data
- *processing*: Rscripts used to generate the dataset
- *vector*: disturbance database for Rhineland-Palatinate
- *aux*: directory for auxiliary data

## Environment Setup

To set up the required R environment, follow these steps:

```sh
conda create -n deep4distR  
conda activate deep4distR 
conda install -c conda-forge r-base=4.3.1 --file requirements.txt
```


## Reproduce the dataset creation

1) Download the raw data (DOP images, Point Clouds and DEM) use:
``` sh	
wget -i *_download_url.txt --no-check-certificate -nc 

```
Note that DOP images are constantly updated and therefore their location in the geobasis-rlp.de(geobasis-rlp.de) server might have changed. Similarly for the BDOM (point cloud) products.

2) Preprocess the downloaded data
 - 2.1 Run the *01_convert_polygons_to_mask.R* script. It rasterizes the disturbance polygons and save them as geotiff files matching the extent of the DOP.
 - 2.2 Run the *02_preprocess_raw_point_clouds.R* script. It extracts the DSM from .laz files and save them as geotiff files with the same extent of the DOP.
 - 2.3 Run the *03_create_nDSM.R* script. It creates the DEM files and calculate the nDSM (object height) and save them as geotiff with the same extent as DOP and DSM.
 - 2.4 Run the *04_split_raster_tiles.R* script. It stacks the RGB-NIR-nDSM and extracts the patches of 500x500x5 as presented in the Deep4Dist dataset.  


## Citation

If you use Deep4Dist in your research, please cite:

```
@dataset{rodriguez_paulino_2025_14884819,
  author       = {Rodríguez-Paulino, Enmanuel and
                  Stoffels, Johannes and
                  Schlerf, Martin and
                  Röder, Achim and
                  Wagner, Alexander and
                  Udelhoven, Thomas},
  title        = {A high-resolution dataset for forest disturbance
                   mapping
                  },
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14884819},
  url          = {https://doi.org/10.5281/zenodo.14884819},
}
```

## License

Deep4Dist dataset is released under the CCBY-4.0 and the code under Apache 2.0 Licenses.

## Contact

For questions or contributions, open an issue in this repository.

