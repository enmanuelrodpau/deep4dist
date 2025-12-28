import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.classification import  JaccardIndex,  Accuracy, F1Score, ConfusionMatrix, CohenKappa
from dataset import Deep4DistDataset
from model import UNetModel
from utils import read_config, print_config
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_metrics(config: dict):
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
                 
    dataset = Deep4DistDataset( 
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
    num_classes=config['num_classes']
    
    masks = []
    images = []
    kappas = []
    metric_vals = []
    oas = []
    metric_df = []
    predictions = []
    paths = []

    metrics = MetricCollection({
                'conf_matrix': ConfusionMatrix(num_classes=num_classes, task='multiclass'),
                'kappa': CohenKappa(num_classes=num_classes, task='multiclass', ignore_index=ignore_index),
                'accuracy': Accuracy(num_classes=num_classes, multidim_average='global', average=None, ignore_index=ignore_index, task='multiclass'),
                'F1Score': F1Score(num_classes=num_classes, multidim_average='global', average=None, ignore_index=ignore_index, task='multiclass'),
                'class_IoU': JaccardIndex(num_classes=num_classes, average=None, ignore_index=ignore_index, task='multiclass')
            }).to(device)

    with torch.no_grad():  
        for batch in tqdm(dataloader):
            img, msk, pths = batch['img'].to(device), batch['msk'].to(device), batch['img_pth']

            preds = model(img)
            preds = torch.argmax(preds, dim=1)
            msk = torch.argmax(msk, dim=1)
            preds = preds[:,:500,:500]

            metrics.update(preds, msk)
            predictions.append(preds.cpu().numpy())
            paths.append(pths)
            
    metrics_dict = metrics.compute()
    conf_matrix = metrics_dict['conf_matrix'].cpu().numpy()

    kappas.append(metrics_dict['kappa'].cpu().numpy().tolist() )
    metric_vals.append({f'mean_{key}': (torch.mean(metrics_dict[key])).cpu().numpy().tolist() for key in ['F1Score', 'accuracy', 'class_IoU']})
    oas.append(np.trace(conf_matrix)/np.sum(conf_matrix))

    metrics.reset()
    val_met = pd.DataFrame(metric_vals)
    val_met['kappa'] = kappas
    val_met['OA'] = oas

    metric_df.append(val_met)
        
    Path(config['metric_path'] ).mkdir(parents=True, exist_ok=True)

    val_met.to_csv(Path(config['metric_path'] )/ f'metrics_baseline.csv')

    met = {'F1Score': metrics_dict['F1Score'].cpu().numpy(), 
       'class_IoU': metrics_dict['class_IoU'].cpu().numpy()}

    met = pd.DataFrame.from_records(met)
    
    met.to_csv(Path(config['metric_path'] )/ f'classwise_metrics_baseline.csv')

    cm = pd.DataFrame.from_records(conf_matrix)
    cm['class'] = ['BG','BB', 'CC', 'WT']
    cm.to_csv(Path(config['metric_path'] )/ f'cm_baseline.csv')
    
    print(val_met.to_markdown(index=False))
    
    print('\n')
    
    print(met.to_markdown(index=False))


    if config['save_predictions']:
        save_dir = Path(config['out_folder'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
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
   
    run_metrics(config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_metrics.py <config_path>')
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
