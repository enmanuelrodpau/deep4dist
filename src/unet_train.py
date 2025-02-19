import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import logging 
import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import MetricCollection, MeanMetric 
from torchmetrics.classification import  JaccardIndex,  Accuracy, F1Score
import segmentation_models_pytorch as smp


import pandas as pd
import datetime
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss

import torchvision.transforms as T
import kornia.augmentation as K


import json
from omegaconf import DictConfig, ListConfig
from rich import get_console
from rich.style import Style
from rich.tree import Tree

from typing import List, Optional, Dict, Mapping, Callable, Tuple, Union

from pathlib import Path
import yaml
import copy
import shutil

torch.cuda.empty_cache() 

def min_max_normalize(tensor, q_low, q_hi):
    q_low = torch.as_tensor(q_low, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    q_hi = torch.as_tensor(q_hi, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    epsilon = 1e-12
    denominator = (q_hi - q_low) + epsilon
    tensor = (tensor - q_low) / denominator
    return tensor

# Dataset Classes
class CustomDataset(Dataset):
    def __init__(self, img_list: List[str], path_to_img: str, path_to_msk: Optional[str] = None,
                 channels: List[int] = [1, 2, 3, 4, 5], num_classes: int = 4,
                 use_augmentations: Optional[K.container.ImageSequential] = None, norm_function: Optional[str] = None,
                 means: Optional[List[float]] = None, stds: Optional[List[float]] = None,
                 q_low: Optional[List[float]] = None, q_hi: Optional[List[float]] = None,
                 padding: bool = False, target_size: List[int] = [512, 512]):
        self.list_imgs = [f"{path_to_img}/{file}.tif" for file in img_list]
        self.list_msks = [f'{path_to_msk}/{file.replace("img","msk")}.tif' for file in img_list] if path_to_msk else None
        self.channels = channels
        self.num_classes = num_classes
        self.use_augmentations = use_augmentations
        self.norm_function = norm_function
        self.means = means
        self.stds = stds
        self.q_low = q_low
        self.q_hi = q_hi
        self.padding = padding
        self.target_size = target_size

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
        return array

    def read_msk(self, raster_file: str) -> Optional[torch.Tensor]:
        if self.list_msks:
            with rasterio.open(raster_file) as src_msk:
                array = src_msk.read(1)
                array = torch.nn.functional.one_hot(torch.as_tensor(array, dtype=torch.long), num_classes=self.num_classes).permute(2, 0, 1)
            return array
        return None

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        img = self.read_img(self.list_imgs[index])
        msk = self.read_msk(self.list_msks[index]) if self.list_msks else None
        img = torch.as_tensor(img, dtype=torch.float)
        if msk is not None:
            msk = torch.as_tensor(msk, dtype=torch.float)

        original_size = list(img.shape[-2:])
        if self.padding:
            img = torch.nn.functional.pad(img, pad=(0, self.target_size[1] - img.shape[-1], 0, self.target_size[0] - img.shape[-2]), mode='constant', value=0)
            

        if self.norm_function == "mean_sd" and self.means and self.stds:
            normalize = T.Normalize(torch.as_tensor(self.means,dtype=img.dtype), torch.as_tensor(self.stds,dtype=img.dtype))
            img = normalize(img)#(img - torch.as_tensor(self.means)[:, None, None]) / torch.as_tensor(self.stds)[:, None, None]
        elif self.norm_function == "quantile" and self.q_low and self.q_hi:
            img = min_max_normalize(img, self.q_low, self.q_hi)
            
        img = img.clip(0,1)
        
        if self.use_augmentations:
            augmentations = self.use_augmentations
            aug_params = augmentations.forward_parameters(img.unsqueeze(0).shape)
            img = (augmentations(img.unsqueeze(0), params = aug_params)).squeeze(0)
            
            if msk is not None:
                aug_params[0][1]['batch_prob'] = torch.as_tensor(0, dtype=torch.float32)
                msk = (augmentations(msk.unsqueeze(0), params = aug_params)).squeeze(0)

        return {"img": img, "msk": msk, "original_size": original_size} if msk is not None else {"img": img}

# DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_list: List[str], val_list: List[str], path_to_img: str, path_to_msk: str,
                 batch_size: int = 2, num_workers: int = 1, use_augmentations: Optional[torch.nn.Sequential] = None,
                 channels: List[int] = [1, 2, 3, 4, 5], num_classes: int = 4, norm_function: Optional[str] = None,
                 means: Optional[List[float]] = None, stds: Optional[List[float]] = None,
                 q_low: Optional[List[float]] = None, q_hi: Optional[List[float]] = None,
                 padding: bool = False, target_size: List[int] = [512, 512]):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.path_to_img = path_to_img
        self.path_to_msk = path_to_msk
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentations = use_augmentations
        self.channels = channels
        self.num_classes = num_classes
        self.norm_function = norm_function
        self.means = means
        self.stds = stds
        self.q_low = q_low
        self.q_hi = q_hi
        self.padding = padding
        self.target_size = target_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.train_list, self.path_to_img, self.path_to_msk, self.channels,
                                               self.num_classes, self.use_augmentations, self.norm_function, self.means,
                                               self.stds, self.q_low, self.q_hi, self.padding, self.target_size)
            self.val_dataset = CustomDataset(self.val_list, self.path_to_img, self.path_to_msk, self.channels,
                                             self.num_classes, None, self.norm_function, self.means, self.stds,
                                             self.q_low, self.q_hi, self.padding, self.target_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last = True)

# Model
class UNetModel(pl.LightningModule):
    def __init__(
        self,
        
        config,
        
        
    ):
        super().__init__()

         
        self.num_classes = config["num_classes"]

        self.config = config


        if self.config["resume_training"]:

            self.model = smp.create_model(
                                            arch=self.config["architecture"],
                                            encoder_name=self.config["encoder"],
                                            classes=self.config['previous_num_classes'],
                                            in_channels=self.config['num_channels_aerial']
                                            )

            self.model = load_model(self.config["weight_pth"], self.model)
            
            

            if self.config["decoder_only"]:

                  for param in self.model.encoder.parameters():
                        param.requires_grad = False


            self.model.segmentation_head = torch.nn.Sequential(
               torch.nn.Conv2d(16, config['num_classes'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
               torch.nn.Identity(),
               torch.nn.Identity()
            )

        else:
        
            self.model= smp.create_model(
                                        arch=self.config["architecture"], 
                                        encoder_name=self.config["encoder"], 
                                        classes=self.config["num_classes"], 
                                        in_channels=self.config["num_channels_aerial"],
                                        encoder_weights= None if self.config["encoder_weights"] == 'None' else self.config["encoder_weights"] 
                                        )
                                        
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])

        scheduler_params = {

        "monitor": self.config["monitor"],
        "mode": self.config["mode"],
        "patience": int(self.config["patience"]),
        "factor": float(self.config["factor"]),

        }

        self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode=scheduler_params["mode"],
        factor=scheduler_params["factor"],
        patience=scheduler_params["patience"]
        )

        with torch.no_grad():
            self.class_weights = torch.FloatTensor(np.array(list(config['weights_aerial_satellite'].values()))[:,0])


        self.criterion = DiceFocalLoss(softmax=True,
                               reduction="mean",
                               include_background=self.config["include_background"],
                               weight = self.class_weights)
        
        
        
        self.ignore_index = None if self.config["include_background"] else int(self.config["index_background"])
        
      
                                             
        metrics = MetricCollection({
            'Accuracy': Accuracy(num_classes=self.num_classes, multidim_average='global', average=None, ignore_index=self.ignore_index, task="multiclass"),
            'F1Score': F1Score(num_classes=self.num_classes, multidim_average='global', average=None, ignore_index=self.ignore_index, task="multiclass"),
            'Class_IoU': JaccardIndex(num_classes=self.num_classes, average=None, ignore_index=self.ignore_index, task="multiclass")
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch["img"], batch["msk"] 
        logits = self(imgs)
        logits = logits[:, :, 0:masks.shape[-2], 0:masks.shape[-1]]
        
        
        
        loss = self.criterion(logits, masks) 
        
        self.train_loss.update(loss)
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
        
        
        masks = torch.argmax(masks, dim=1)
        self.train_metrics.update(preds.detach(), masks)
        
        
        return {"loss": loss, "preds": preds.detach()}
        
    def on_after_backward(self):
        
        with open(f'unused_parameters_{self.config["out_model_name"]}.txt', "w") as f:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is None:
                    f.write(f"{name}\n")
    
    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        

        output = self.train_metrics.compute()
        self.log_dict({"train_loss": train_loss.item(),
        
                   "train_mean_IoU_BG": torch.mean(output["train_Class_IoU"]),
                   "train_meanF1Score_BG": torch.mean(output["train_F1Score"]),
                   "train_meanAccuracy_BG": torch.mean(output["train_Accuracy"]), 
                   
                   "train_mean_IoU": torch.mean(output["train_Class_IoU"][1:]),
                   "train_meanF1Score": torch.mean(output["train_F1Score"][1:]),
                   "train_meanAccuracy": torch.mean(output["train_Accuracy"][1:]),
                   
                   "train_F1Score_BG" : output["train_F1Score"][0],
                   "train_F1Score_BB" : output["train_F1Score"][1],
                   "train_F1Score_CC" : output["train_F1Score"][2],
                   "train_F1Score_WT" : output["train_F1Score"][3],

                   "train_IoU_BG" : output["train_Class_IoU"][0], 
                   "train_IoU_BB" : output["train_Class_IoU"][1],
                   "train_IoU_CC" : output["train_Class_IoU"][2],
                   "train_IoU_WT" : output["train_Class_IoU"][3],

                   "train_Acc_BG" : output["train_Accuracy"][0], 
                   "train_Acc_BB" : output["train_Accuracy"][1],
                   "train_Acc_CC" : output["train_Accuracy"][2],
                   "train_Acc_WT" : output["train_Accuracy"][3],

        }, sync_dist=True, on_epoch=True, logger=True)

        self.train_metrics.reset()
        self.train_loss.reset()



    def validation_step(self, batch, batch_idx):
        imgs, masks = batch["img"], batch["msk"] 
        logits = self(imgs)
        logits = logits[:, :, 0:masks.shape[-2], 0:masks.shape[-1]]
        
        loss = self.criterion(logits, masks)
        
        self.val_loss.update(loss)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
        
        masks = torch.argmax(masks, dim=1)
        
        self.val_metrics.update(preds.detach(), masks)
        
        return {"loss": loss, "preds": preds.detach()}
        
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
       
        
        
        output = self.val_metrics.compute()
        
        self.log_dict({"val_loss": val_loss.item(),
                       "val_mean_IoU_BG": torch.mean(output["val_Class_IoU"]),
                       "val_meanF1Score_BG": torch.mean(output["val_F1Score"]),
                       "val_meanAccuracy_BG": torch.mean(output["val_Accuracy"]),
                       
                       "val_mean_IoU": torch.mean(output["val_Class_IoU"][1:]),
                       "val_meanF1Score": torch.mean(output["val_F1Score"][1:]),
                       "val_meanAccuracy": torch.mean(output["val_Accuracy"][1:]),
                       
                       
                       "val_F1Score_BG" : output["val_F1Score"][0],
                       "val_F1Score_BB" : output["val_F1Score"][1],
                       "val_F1Score_CC" : output["val_F1Score"][2],
                       "val_F1Score_WT" : output["val_F1Score"][3],
            
                       "val_IoU_BG" : output["val_Class_IoU"][0],
                       "val_IoU_BB" : output["val_Class_IoU"][1],
                       "val_IoU_CC" : output["val_Class_IoU"][2],
                       "val_IoU_WT" : output["val_Class_IoU"][3],
            
                       "val_Acc_BG" : output["val_Accuracy"][0],
                       "val_Acc_BB" : output["val_Accuracy"][1],
                       "val_Acc_CC" : output["val_Accuracy"][2],
                       "val_Acc_WT" : output["val_Accuracy"][3],
            
            }, sync_dist=True, on_epoch=True, logger=True)
        
        self.val_metrics.reset()
        self.val_loss.reset()

        
        
       

    def configure_optimizers(self):
        if self.scheduler is not None:
            
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                   mode=self.config["mode"], 
                                                                   factor=float(self.config["factor"]), 
                                                                   patience=int(self.config["patience"]))
             
            config_ = {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.config["monitor"]
                }
            }
            return config_
        else: return self.optimizer




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
    
@rank_zero_only
def print_model(config):
    print(f'Starting model {config["architecture"]} with encoder {config["encoder"]}')



# Cross-Validation Training and Averaging Model Weights
def run_cross_validation(img_list: List[str], path_to_img: str, path_to_msk: str, spatial_blocks: List[List[int]], config: dict):
    fold = 0
    
        
    splits = pd.read_csv("/data/data_splits_FOA_c.csv")
    splits = (splits.groupby('split')['tile_name'].apply(list)).tolist()
    train_list = [f"{tile}/vhr/img{tile[-8:]}" for tile in splits[1]]
    val_list = [f"{tile}/vhr/img{tile[-8:]}" for tile in splits[2]]
    test_list = [f"{tile}/vhr/img{tile[-8:]}" for tile in splits[0]]

    use_augmentations = K.container.ImageSequential(
                K.RandomChannelDropout(num_drop_channels=2, fill_value=0.0, p=0.5, same_on_batch =False),
                K.RandomVerticalFlip(p=0.5, same_on_batch =False),
                K.RandomHorizontalFlip(p=0.5, same_on_batch = False),
                K.RandomRotation(degrees=90.0, p=0.5, same_on_batch =False )
            )

    data_module = CustomDataModule(train_list,
                                   val_list,
                                   path_to_img,
                                   path_to_msk,
                                   batch_size= int(config["batch_size"]),
                                   num_workers= int(config["num_workers"]),
                                   num_classes= int(config["num_classes"]),
                                   norm_function = str(config["Norm_function"]),
                                   use_augmentations=use_augmentations,
                                   q_low= config["q_low"],
                                   q_hi=config["q_hi"],
                                   padding=  True if config['vhr_hw'] < config["target_size"] else False,
                                   target_size=config["target_size"])
                                   


    

    
    seg_module = UNetModel(
       
        config=config
    )


    # Set up TensorBoard logger
    logger = TensorBoardLogger(save_dir=f"{config['model_dir']}/logs", name=f"fold_{fold + 1}")


                                             
    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    ckpt_callback = ModelCheckpoint(
        monitor="val_meanF1Score",
        dirpath= f"{config['model_dir']}/checkpoints/fold_{fold + 1}",
        filename="best_checkpoint_{epoch}_{val_meanF1Score:.2f}_{val_loss:.2f}",
        save_top_k=-1,  # Save all checkpoints without deleting
        mode="max",
        save_weights_only=False, 
        save_last = True,
        every_n_epochs = 1,
        auto_insert_metric_name=True,
        enable_version_counter = True
    )

    ckpt_callback.CHECKPOINT_NAME_LAST = "epoch_{epoch}_last"


    early_stop_callback = EarlyStopping(
        monitor="val_meanF1Score",
        min_delta=0.1,
        patience=config["early_stopping_patience"],  
        mode="max",
        stopping_threshold=None,
        strict= False,
    )

    

    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        lr_monitor,
  
    ]


    trainer = Trainer(
        precision=16, 
        max_epochs=config["num_epochs"],
        accelerator="auto",
        devices=config["devices"],  
        strategy= config["strategy"],
        num_nodes=config["num_nodes"],  
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        deterministic=config["deterministic_mode"],
        log_every_n_steps=25
    )


        
    trainer.fit(seg_module, datamodule=data_module)#,



# Main

def main(config_path):
    
    config =  read_config(config_path)
    
    arch_list = config["architecture"]
    encoder_list = config["encoder"]
    
    for arch in arch_list:
        config["architecture"] = arch
        
        for encoder in encoder_list:  
            
            config["encoder"] = encoder
            seed_everything(config["seed"], workers = True)
            
            #config["strategy"] = 'auto'
            config["devices"] =  int(os.environ["SLURM_GPUS_ON_NODE"])
            config["num_nodes"] =  int(os.environ["SLURM_NNODES"])
            config["batch_size"] = config["batch_size"] 
            
            print_config(config)
            
            print_model(config)  

            timestamp = datetime.datetime.now().strftime('%Y%m%d/%H%M%S')
            model_name = f"{config['out_model_name']}_{config['architecture']}_{config['encoder']}_{config['num_epochs']}_es_{config['early_stopping_patience']}_bs_{config['batch_size']}"
            out_dir = Path(config["out_folder"], model_name, timestamp)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(config_path, out_dir)
            shutil.copy(__file__, out_dir) #backup script
            
            pth_in = Path(config['path_data'])

            tile_names = [tile.parts[-1] for tile in list(pth_in.glob("*"))]
            pth_list = [pth_in / tile / "meta" for tile in tile_names ]
                 
            meta_list = [list(pth.glob("*.json"))[0].as_posix() for pth in pth_list]
            splt = []
            kfolds = []
            img_names = []
            msk_names = []

            for pth in meta_list:    
              with open(pth) as f:
                  meta = json.load(f)
                  splt.append(meta["split_meta"]["split"][0])
                  kfolds.append(meta["split_meta"]["k_fold"][0])
                  img_names.append(meta["vhr_meta"]["vhr_img"][0])
                  msk_names.append(meta["vhr_meta"]["vhr_msk"][0])
                  
                
            data_dict = {
                "tile_name" : tile_names,
                "img" : img_names,
                "msk" : msk_names,
                "k_fold" : kfolds,
                "split" : splt,
            }
            
            img_list_kfolds = pd.DataFrame.from_dict(data_dict)

            img_list_kfolds["idx"] = img_list_kfolds.index
            img_list = [f'{tile}/vhr/{img}' for tile,img in zip(tile_names,img_names)]

            spatial_blocks = (img_list_kfolds.groupby('k_fold')['idx'].apply(list)).tolist()
            spatial_blocks = [
                (spatial_blocks[0] + spatial_blocks[2] + spatial_blocks[3], spatial_blocks[4], spatial_blocks[1]), #1,3,4..,5
                (spatial_blocks[2] + spatial_blocks[3] + spatial_blocks[4], spatial_blocks[0], spatial_blocks[1]), # 3,4,5..,1
                (spatial_blocks[0] + spatial_blocks[3] + spatial_blocks[4], spatial_blocks[2], spatial_blocks[1]),#1,4,5,3
                (spatial_blocks[0] + spatial_blocks[2] + spatial_blocks[4], spatial_blocks[3], spatial_blocks[1]),#1,3,5,4
                
            ]
            
            
            config["model_dir"] = out_dir
            
            config["out_model_name"] = model_name
           
            run_cross_validation(img_list, 
                          config["path_data"], 
                          config["path_data"], 
                          spatial_blocks, 
                          config)







# Run main
if __name__ == "__main__":
    config_path = '/config/unet.yml'
    main(config_path)