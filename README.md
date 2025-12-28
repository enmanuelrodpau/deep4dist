# Deep Learning For Forest Disturbance mapping (Deep4Dist)
This repository hosts the code and notebooks used in the article A high-resolution dataset for forest disturbance mapping*.

## Overview

The Deep4Dist dataset is a novel benchmark dataset comprising approximately 17,500 georeferenced image patches extracted from high-resolution digital orthophotos of Rhineland-Palatinate, Germany. Each 500 × 500 pixel image (at 20 cm resolution) includes five spectral channels (RGB, near-infrared, and normalized digital surface model), which together capture both spectral and structural information critical for distinguishing among disturbance types such as bark beetle damage, clear-cuts, and windthrow events.

The dataset has been meticulously curated with high-quality annotations and detailed metadata, ensuring its reliability and facilitating integration with medium-resolution satellite data.

## Features

- **Multispectral Data:** Includes high-resolution digital orthophotos with five spectral channels (RGB, near-infrared, and normalized digital surface model).
- **Temporal Component:** Designed for integration with medium-resolution satellite time series data.
- **High-Resolution Labels:** Ground-based and expert-annotated forest disturbances for accurate model training.
- **Semantic Segmentation:** Designed for pixel-wise classification using deep learning methods.

## Repository Structure

This repository is organized into dedicated branches to clearly separate dataset construction from benchmark experimentation:

- **dataset branch[https://github.com/enmanuelrodpau/deep4dist/tree/main/dataset]:** Contains all files, scripts, and metadata required to re-create the Deep4Dist dataset, including data preprocessing and annotation-related resources.

- **benchmark branch[https://github.com/enmanuelrodpau/deep4dist/tree/main/benchmark]:** Contains the code and configuration files necessary to run the benchmark experiments, including model training, evaluation pipelines, and reproducibility settings.

## Downloading the Dataset

The dataset is openly available at: [Zenodo](https://zenodo.org/records/14884819)

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
