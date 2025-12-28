import os
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
from pathlib import Path
import pytorch_lightning as pl

def min_max_normalize(tensor, q_low, q_hi):
    q_low = torch.as_tensor(q_low, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    q_hi = torch.as_tensor(q_hi, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    epsilon = 1e-12
    denominator = (q_hi - q_low) + epsilon
    tensor = (tensor - q_low) / denominator
    return tensor

class Deep4DistDataset(Dataset):
    def __init__(self, 
                 img_list, 
                 msk_list=None, 
                 channels=[1, 2, 3, 4, 5], 
                 num_classes=4,
                 use_augmentations=None, 
                 norm_function=None, 
                 means=None, 
                 stds=None,
                 q_low=None, 
                 q_hi=None, 
                 padding=False, 
                 target_size=[512, 512]):
        self.list_imgs = img_list
        self.list_msks = msk_list
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
    def read_img(self, raster_file):
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
        return array
    def read_msk(self, raster_file):
        if self.list_msks:
            with rasterio.open(raster_file) as src_msk:
                array = src_msk.read(1)
                array = torch.nn.functional.one_hot(
                    torch.as_tensor(array, dtype=torch.long), num_classes=self.num_classes
                ).permute(2, 0, 1)
            return array
        return None
    def __len__(self):
        return len(self.list_imgs)
    def __getitem__(self, index):
        img_pth = self.list_imgs[index]
        img = self.read_img(self.list_imgs[index])
        msk = self.read_msk(self.list_msks[index]) if self.list_msks else None
        img = torch.as_tensor(img, dtype=torch.float)
        if msk is not None:
            msk = torch.as_tensor(msk, dtype=torch.float)
        
        if self.padding:
            img = F.pad(
                img,
                pad=(0, self.target_size[1] - img.shape[-1], 0, self.target_size[0] - img.shape[-2]),
                mode='constant',
                value=0
            )
        if self.norm_function == 'mean_sd' and self.means and self.stds:
            normalize = T.Normalize(
                torch.as_tensor(self.means, dtype=img.dtype),
                torch.as_tensor(self.stds, dtype=img.dtype)
            )
            img = normalize(img)
        elif self.norm_function == 'quantile' and self.q_low and self.q_hi:
            img = min_max_normalize(img, self.q_low, self.q_hi)
        img = img.clamp(0, 1)
        if self.use_augmentations:
            augmentations = self.use_augmentations
            aug_params = augmentations.forward_parameters(img.unsqueeze(0).shape)
            img = (augmentations(img.unsqueeze(0), params = aug_params)).squeeze(0)
            if msk is not None:
                aug_params[0][1]['batch_prob'] = torch.as_tensor(0, dtype=torch.float32)
                msk = (augmentations(msk.unsqueeze(0), params = aug_params)).squeeze(0)
        
        return {'img': img, 'msk': msk, 'img_pth': img_pth} if msk is not None else {'img': img, 'img_pth': img_pth}

class Deep4DistDM(pl.LightningDataModule):
    def __init__(self, train_list, val_list, batch_size=2, num_workers=1, use_augmentations=None,
                 channels=[1, 2, 3, 4, 5], num_classes=4, norm_function=None,
                 means=None, stds=None, q_low=None, q_hi=None, padding=False, target_size=[512, 512]):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
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
            self.train_dataset = Deep4DistDataset(
                img_list = self.train_list[0], 
                msk_list = self.train_list[1], 
                channels = self.channels, 
                num_classes = self.num_classes,
                use_augmentations = self.use_augmentations, 
                norm_function = self.norm_function, 
                means = self.means, 
                stds = self.stds,
                q_low = self.q_low, 
                q_hi = self.q_hi, 
                padding = self.padding, 
                target_size = self.target_size
            )
            self.val_dataset = Deep4DistDataset(
                img_list = self.val_list[0], 
                msk_list = self.val_list[1], 
                channels = self.channels, 
                num_classes = self.num_classes,
                use_augmentations = None, 
                norm_function = self.norm_function, 
                means = self.means, 
                stds = self.stds,
                q_low = self.q_low, 
                q_hi = self.q_hi, 
                padding = self.padding, 
                target_size = self.target_size
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )
