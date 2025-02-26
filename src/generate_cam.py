import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from dataset import CustomDataset
from model import UNetModel
from utils import read_config, print_config, save_mask
from torch.utils.data import DataLoader
from tqdm import tqdm

module_path = str("./Seg_XRes_CAM")
if module_path not in sys.path:
    sys.path.append(module_path)
from seg_xres_cam import *
from visualize import visualize_algos
from PIL import Image
from itertools import combinations

def camp_band_selection(cam_name: str, bands: list):
    if cam_name == 'baseline':
        return None
    elif cam_name == 'single':
        return bands
    elif cam_name == 'multi':
        return list(combinations(bands, 2))
    else:
        raise ValueError(f"Unknown camera type: {cam_name}")
    


def run_cam(config: dict):
    print_config(config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = device

    method_dict = {'Seg-Grad-CAM': 0, 'Seg-XRes-CAM': 1}
    method_indexes = method_dict['Seg-XRes-CAM']

    pool_sizes, pool_modes, reshape_transformer = 1, np.max, False
    fig_size = (30, 50)
    vis, vis_base, vis_rise, grid = False, False, False, True
    preprocess_transform = None
    
    use_augmentations = None
    
    ignore_index = None if config['include_background'] else int(config['index_background'])
    
    num_classes=int(config['num_classes'])
    
    band_names = config['band_names']
    n_bands = camp_band_selection(config['cam_type'], list(range(config['num_channels_aerial'])))
    
    splits = pd.read_csv(config['path_metadata'])
    splits = (splits.groupby('split')['tile_name'].apply(list)).tolist()
    
    path_to_data = Path(config['path_data'])
        
    test_list = [[(path_to_data/'test'/'image'/f'{tile}.tif').as_posix() for tile in splits[0]],
    [(path_to_data/'test'/'mask'/f'{tile}.tif').as_posix() for tile in splits[0]]]
    
    save_dir = Path(config['out_folder']) / config['cam_type']
    
    dataset = CustomDataset( 
                 img_list = test_list[0], 
                 msk_list=test_list[1], 
                 channels=list(range(1, config['num_channels_aerial']+1)),
                 num_classes= num_classes,
                 norm_function = str(config['Norm_function']),
                 use_augmentations=use_augmentations,
                 q_low= config['q_low'],
                 q_hi=config['q_hi'],
                 padding=True, 
                 target_size=[512, 512])
                 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last = False)
    
    model = UNetModel(
       
        config=config
    )
    
    ckpt_path = config['weight_pth']


    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage) 

    model.load_state_dict(checkpoint['state_dict'], strict = False)
    
    model = model.to(device)
    model = model.eval()
    num_classes=config['num_classes']
    
    target_layer = model.model.encoder.layer4


    for batch in tqdm(dataloader):

      if n_bands is None:
        c_map = np.zeros((num_classes, 500,500), dtype =np.float32)
        input_tensor = batch["img"].to("cuda")

        img_pth = batch["img_pth"][0]

        for category in range(num_classes):
            cam_map, _ = seg_grad_cam(input_tensor[0],
                  model,
                  preprocess_transform,
                  target=category,
                  target_layer=target_layer,
                  box=None,
                  DEVICE=DEVICE,
                  method_index=method_indexes,
                  fig_base_name=None,
                  fig_name=None,
                  vis_base=vis_base,
                  vis=vis,
                  negative_gradient=False,
                  pool_size=pool_sizes,
                  pool_mode=pool_modes,
                  reshape_transformer=reshape_transformer)

            c_map[category,::] = cam_map[:500,:500]

        if config['save_cam']:
             
            save_dir.mkdir(parents=True, exist_ok=True)

            msk_pth = Path(save_dir / f'{(Path(img_pth).parts)[-1]}')
            save_mask(c_map, img_pth,msk_pth,num_classes)

      else:

        for band_index in n_bands:

            c_map = np.zeros((num_classes, 500,500), dtype =np.uint8)
            input_tensor = batch["img"].to("cuda")

            img_pth = batch["img_pth"][0]


            if isinstance(band_index, int):
                band = band_names[band_index]
                tmp_save_dir = save_dir / band

                input_tensor[:, band_index, :, :] = 0
            else:
                
                band_comb = "_".join([ band_names[xs] for xs in band_index ])
                tmp_save_dir = save_dir / band_comb

                tmp_save_dir.mkdir(parents=True, exist_ok=True)

                for b in band_index:
                    input_tensor[:, b, :, :] = 0



            for category in range(num_classes):
                cam_map, _ = seg_grad_cam(input_tensor[0],
                     model,
                     preprocess_transform,
                     target=category,
                     target_layer=target_layer,
                     box=None,
                     DEVICE=DEVICE,
                     method_index=method_indexes,
                     fig_base_name=None,
                     fig_name=None,
                     vis_base=vis_base,
                     vis=vis,
                     negative_gradient=False,
                     pool_size=pool_sizes,
                     pool_mode=pool_modes,
                     reshape_transformer=reshape_transformer)

                c_map[category,::] = cam_map[:500,:500]
                
            if config['save_cam']:
                tmp_save_dir.mkdir(parents=True, exist_ok=True)

                msk_pth = Path(tmp_save_dir / f'{(Path(img_pth).parts)[-1]}')
                save_mask(c_map, img_pth,msk_pth,num_classes)


def main(config_path:str):
        
    config =  read_config(config_path)
   
    run_cam(config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python generate_cam.py <config_path>')
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)