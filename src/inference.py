import os
import sys
import pandas as pd
import datetime
import shutil
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset import Deep4DistDataset
from model import UNetModel
from utils import read_config, print_config
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_inference(config: dict):
    
    splits = pd.read_csv(config["path_metadata"])
    splits = (splits.groupby('split')['tile_name'].apply(list)).tolist()
    
    path_to_data = Path(config["path_data"])
        
    inference_list = [[path_to_data/"test"/"image"/f"{tile}.tif" for tile in splits[0]],
    None]
    
    use_augmentations = None
    
    num_classes=int(config["num_classes"])
                 
    dataset = Deep4DistDataset( 
                 img_list = inference_list[0], 
                 msk_list=inference_list[1], 
                 channels=[1, 2, 3, 4, 5],
                 num_classes= num_classes,
                 norm_function = str(config["Norm_function"]),
                 use_augmentations=use_augmentations,
                 q_low= config["q_low"],
                 q_hi=config["q_hi"],
                 padding=True, 
                 target_size=[512, 512])
                 
    dataloader = DataLoader(dataset, batch_size=config["batch_size_inference"], shuffle=False, num_workers=1, drop_last = False)
    
    model = UNetModel(
       
        config=config
    )
   
   model.load_state_dict(config['weight_pth'], strict = False)
   
   model = model.to(device)
   model = model.eval()
   
   predictions = []
   paths = []
   
   save_dir = Path(config['out_folder']) / config['out_model_name']
   save_dir.mkdir(parents=True, exist_ok=True)
   
    with torch.no_grad():  
        for batch in tqdm(dataloader):
            img, pths = batch["img"].to(device), batch["img_pth"]
            preds = model(img)
            preds = torch.argmax(preds, dim=1)
            msk = torch.argmax(msk, dim=1)
            preds = preds[:,:500,:500]
            metrics.update(preds, msk)
            predictions.append(preds.cpu().numpy())
            paths.append(pths)


    

    all_preds = np.reshape( np.concatenate(predictions, axis=0), (1,len(test_list[0]),1,500,500))
    
    pth_list =  [
                x
                for xs in paths
                for x in xs
            ]
    for i,p in enumerate(pth_list):
        msk = all_preds[0,i,::]
        img_pth = p
        msk_pth = Path(save_dir / f'{(Path(pth_list[i]).parts)[-1]}')
        save_mask(msk, img_pth,msk_pth,1)

def main(config_path:str):
        
    config =  read_config(config_path)
   
    run_inference(config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inference.py <config_path>')
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
