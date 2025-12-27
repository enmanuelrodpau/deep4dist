# Deep Learning For Forest Disturbance mapping (Deep4Dist)
This repository hosts the code used to create the dataset in the article A high-resolution dataset for forest disturbance mapping*.

## Code Structure
*
*
*


## Downloading the Dataset

The dataset is openly available at: [Zenodo](https://zenodo.org/records/14884819)

## Environment Setup

To set up the required R environment, follow these steps:

```sh
conda create -n deep4dist -c conda-forge python=3.11.6 
conda activate deep4dist 
git clone git@github.com:enmanuelrodpau/.git
cd deep4dist 
pip install -r requirements.txt
```

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
