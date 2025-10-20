# Deep Learning For Forest Disturbance mapping (Deep4Dist)
This repository hosts the code and notebooks used in the article A high-resolution dataset for forest disturbance mapping*.

## Overview

The Deep4Dist dataset is a novel benchmark dataset comprising approximately 17,500 georeferenced image patches extracted from high-resolution digital orthophotos of Rhineland-Palatinate, Germany. Each 500 × 500 pixel image (at 20 cm resolution) includes five spectral channels (RGB, near-infrared, and normalized digital surface model), which together capture both spectral and structural information critical for distinguishing among disturbance types such as bark beetle damage, clear-cuts, and windthrow events.

The dataset has been meticulously curated with high-quality annotations and detailed metadata, ensuring its reliability and facilitating integration with medium-resolution satellite data.

## Baseline model

To validate the dataset’s utility, we conducted a study using a U-Net architecture enhanced by multi-stage transfer learning. The validation results demonstrate strong performance (overall accuracy of 88.2%, Macro F1-score of 81.9%, and Macro IoU of 70.3%), underscoring the dataset’s potential for operational forest disturbance mapping.

### Pretrained Model

| Model       | Parameters | Checkpoint                                                          |
|------------|------------|-------------------------------------------------------------------|
| ResU-net-34 | 24.4M      | [Download](https://huggingface.co/enmanuelrp/Deep4Dist-ResU-net-34/) |

## Features

- **Multispectral Data:** Includes high-resolution digital orthophotos with five spectral channels (RGB, near-infrared, and normalized digital surface model).
- **Temporal Component:** Designed for integration with medium-resolution satellite time series data.
- **High-Resolution Labels:** Ground-based and expert-annotated forest disturbances for accurate model training.
- **Semantic Segmentation:** Designed for pixel-wise classification using deep learning methods.

## Dataset Structure

The dataset consists of:

- **High-Resolution Digital Orthophotos:** 500x500 pixel images at 20cm resolution.
- **Annotated Masks:** Ground-truth labels indicating different disturbance types.
- **Metadata:** Detailed descriptions of each image patch.

```
Deep4Dist/
├── train/
│   ├── image/
│   ├── mask/
├── validation/
│   ├── image/
│   ├── mask/
├── test/
│   ├── image/
│   ├── mask/
├── metadata.csv
├── tile_geometries.gpkg
└── README.md
```

## Downloading the Dataset

The dataset is openly available at: [Zenodo](https://zenodo.org/records/14884819)

## Environment Setup

To set up the required Python environment, follow these steps:

```sh
conda create -n deep4dist -c conda-forge python=3.11.6 
conda activate deep4dist 
git clone git@github.com:enmanuelrodpau/.git
cd deep4dist 
pip install -r requirements.txt
```

## Reproducing Experiments

The repository provides all necessary code and configurations to reproduce the experiments presented in the dataset paper. Each experiment has its own configuration YAML file located in the `config` folder. The following commands can be used to reproduce different stages of the experiments:

- **Train a model:**
  ```sh
  python src/train.py ./config/train.yml
  ```
- **Evaluate test set metrics:**
  ```sh
  python src/test_metrics.py ./config/test_metrics.yml
  ```
- **Perform channel ablation analysis:**
  ```sh
  python src/ablation_metrics.py ./config/ablation_metrics.yml
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
