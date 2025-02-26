import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.classification import  JaccardIndex,  Accuracy, F1Score, ConfusionMatrix, CohenKappa

from dataset import CustomDataset
from model import UNetModel
from utils import read_config, print_config
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
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

def feature_ablation(model, dataloader, band_index, ignore_index, num_classes):
    total_accuracy = []
    metrics = MetricCollection({
                "conf_matrix": ConfusionMatrix(num_classes=num_classes, task="multiclass"),
                'kappa': CohenKappa(num_classes=num_classes, task="multiclass", ignore_index=ignore_index),    
                'accuracy': Accuracy(num_classes=num_classes, multidim_average='global', average=None, ignore_index=ignore_index, task="multiclass"),
                'F1Score': F1Score(num_classes=num_classes, multidim_average='global', average=None, ignore_index=ignore_index, task="multiclass"),
                'class_IoU': JaccardIndex(num_classes=num_classes, average=None, ignore_index=ignore_index, task="multiclass")
            }).to(model.device)
    
    for batch in tqdm(dataloader):
        input_tensor = batch["img"].to(model.device).clone()
        target_tensor = batch["msk"].argmax(dim=1).to(model.device)
        
        if isinstance(band_index, int):
            input_tensor[:, band_index, :, :] = 0
        else:
            for b in band_index:
                input_tensor[:, b, :, :] = 0
   
        
        with torch.no_grad():
            output = model(input_tensor).argmax(dim=1)
        metrics.update(output[:,:500,:500], target_tensor)
        
        
    
    return metrics.cpu().compute()




def run_ablation(config: dict):
    print_config(config)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    splits = pd.read_csv(config['path_metadata'])
    splits = (splits.groupby('split')['tile_name'].apply(list)).tolist()
    
    path_to_data = Path(config['path_data'])
        
    test_list = [[(path_to_data/'test'/'image'/f'{tile}.tif').as_posix() for tile in splits[0]],
    [(path_to_data/'test'/'mask'/f'{tile}.tif').as_posix() for tile in splits[0]]]
    
    use_augmentations = None
    
    ignore_index = None if config['include_background'] else int(config['index_background'])
    
    num_classes=int(config['num_classes'])
                 
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
                 
    dataloader = DataLoader(dataset, batch_size=config['batch_size_inference'], shuffle=False, num_workers=config['num_workers'], drop_last = False)
    
    model = UNetModel(
       
        config=config
    )
    
    ckpt_path = config['weight_pth']


    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage) 

    model.load_state_dict(checkpoint['state_dict'], strict = False)
    
    model = model.to(device)
    model = model.eval()
    
    band_names = config['band_names']
        
    mult_band_ablation_results = [
    
    feature_ablation(model, dataloader, band, ignore_index, num_classes)
    for band in camp_band_selection('multi', list(range(config['num_channels_aerial']))) 
    ]
    
    single_ablation_results = [
        feature_ablation(model, dataloader, band, ignore_index, num_classes)
        for band in camp_band_selection('single', list(range(config['num_channels_aerial']))) 
    ]
    
    Path(config['metric_path'] ).mkdir(parents=True, exist_ok=True)
    
    m_ablation_results_dict = [{key: value.numpy().tolist()  for key,value in mt.items() } for mt in mult_band_ablation_results]
    s_ablation_results_dict = [{key: value.numpy().tolist()  for key,value in mt.items() } for mt in single_ablation_results]
    
    band_comb =  [ f'{band_names[xs[0]]}_{band_names[xs[1]]}' 
    for xs in camp_band_selection('multi', list(range(config['num_channels_aerial']))) 
       
    ]

    path = Path(config['metric_path'] )
    path.mkdir(parents=True, exist_ok=True)
    
    # single_band_feature_ablation
    df = []
    for i in range(len(band_names)):
        tmp = pd.DataFrame.from_records(s_ablation_results_dict[i])
        tmp["class"] = ["BG","BB", "CC", "WT"]
        tmp["band"] = band_names[i]
        
        df.append(tmp )
        pd.concat(df).to_csv((path / f"single_band_feature_ablation.csv").as_posix())


    # multi_feature_ablation    
    tmp_dict = deepcopy(m_ablation_results_dict)

    df = []
    for i in range(len(tmp_dict)):
        tmp = pd.DataFrame.from_records(tmp_dict[i].pop("conf_matrix"))
        tmp["class"] = ["BG","BB", "CC", "WT"]
        tmp["band"] = band_comb[i]
        
        df.append(tmp )
        pd.concat(df).to_csv((path  / f"multi_feature_ablation.csv").as_posix())
        
        
    # multi_cm_feature_ablation
    tmp_dict = deepcopy(m_ablation_results_dict)

    df = []
    for i in range(len(tmp_dict)):
        tmp = pd.DataFrame.from_records(tmp_dict[i]["conf_matrix"])
        tmp["class"] = ["BG","BB", "CC", "WT"]
        tmp["band"] = band_comb[i]
        
        df.append(tmp )
        pd.concat(df).to_csv((path  / f"multi_cm_feature_ablation.csv").as_posix())
        
    # single_cm_feature_ablation
    df = []
    for i in range(len(s_ablation_results_dict)):
        tmp = pd.DataFrame.from_records(s_ablation_results_dict[i]["conf_matrix"])
        tmp["class"] = ["BG","BB", "CC", "WT"]
        tmp["band"] = band_comb[i]
        
        df.append(tmp )
        pd.concat(df).to_csv((path  / f"single_cm_feature_ablation.csv").as_posix())

def main(config_path:str):
        
    config =  read_config(config_path)
   
    run_ablation(config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ablation_metrics.py <config_path>')
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)