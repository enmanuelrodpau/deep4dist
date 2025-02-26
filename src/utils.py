import yaml
import logging 
import os
import torch
from omegaconf import DictConfig, ListConfig
from rich import get_console
from rich.style import Style
from rich.tree import Tree
from typing import Mapping
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
import rasterio

#### LOGGERS 
LOGGER = logging.getLogger(__name__)
log = logging.getLogger('stdout_detection')
log.setLevel(logging.DEBUG)
STD_OUT_LOGGER = log
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
STD_OUT_LOGGER.addHandler(ch)
# Some functions to load the model. Taken from: https://github.com/IGNF/FLAIR-1/blob/main/src/zone_detect/model.py
def get_module(checkpoint: str | Path) -> Mapping:
    if checkpoint is not None and Path(checkpoint).is_file():
        weights = torch.load(checkpoint, map_location='cpu')
        if checkpoint.endswith('.ckpt'):
            weights = weights['state_dict']
    else:
        LOGGER.error('Error with checkpoint provided: either a .ckpt with a "state_dict" key or an OrderedDict pt/pth file')
        return {}
    if 'model.seg_model' in list(weights.keys())[0]:
        weights = {k.partition('model.seg_model.')[2]: v for k, v in weights.items()}
        weights = {k: v for k, v in weights.items() if k != ""}
    return weights
#
def load_model(path, model) -> torch.nn.Module:
    checkpoint = path
    state_dict = get_module(checkpoint=checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model
##########----- Utils Yaml -----#########
def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
@rank_zero_only
def print_config(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.
    Args: config: Config to print to console using a Rich tree.
    """
    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                #print('HERE', group_name)
                branch = tree.add(str(group_name), style=Style(color='yellow', bold=True))
                walk_config(branch, group_option)
            elif isinstance(group_option, ListConfig):
                if not group_option:
                    tree.add(f'{group_name}: []', style=Style(color='yellow', bold=True))
                else:
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='yellow', bold=True))
            else:
                if group_name == '_target_':
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='white', italic=True, bold=True))
                else:
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='yellow', bold=True))
    tree = Tree(
        ':deciduous_tree: Configuration Tree ',
        style=Style(color='white', bold=True, encircle=True),
        guide_style=Style(color='bright_green', bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    get_console().print(tree)
 
def save_mask(mask,  image, image_save, bands):
    with rasterio.open(image) as src:
        transform = src.transform
        crs = src.crs
        r_dtype = mask.dtype

    with rasterio.open(
        image_save,
        "w",
        driver="GTiff",
        height=mask.shape[1],
        width=mask.shape[2],
        count=bands,
        dtype=r_dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(bands):
            dst.write(mask[band, :, :], band + 1)
 