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

from dataset import DataModule
from model import UNetModel
from utils import read_config, print_config
import kornia.augmentation as K

def run_train(config: dict):
    
    splits = pd.read_csv(config["path_metadata"])
    splits = (splits.groupby('split')['tile_name'].apply(list)).tolist()
    
    path_to_data = Path(config["path_data"])
    
    train_list = [[(path_to_data/"train"/"image"/f"{tile}.tif").as_posix() for tile in splits[1]],
    [(path_to_data/"train"/"mask"/f"{tile}.tif").as_posix() for tile in splits[1]]]
    
    val_list = [[(path_to_data/"validation"/"image"/f"{tile}.tif").as_posix() for tile in splits[2]],
    [(path_to_data/"validation"/"mask"/f"{tile}.tif").as_posix() for tile in splits[2]]]
        
    use_augmentations = K.container.ImageSequential(
                K.RandomChannelDropout(num_drop_channels=2, fill_value=0.0, p=0.5, same_on_batch =False),
                K.RandomVerticalFlip(p=0.5, same_on_batch =False),
                K.RandomHorizontalFlip(p=0.5, same_on_batch = False),
                K.RandomRotation(degrees=90.0, p=0.5, same_on_batch =False )
                
                
            )
            
                 
    data_module = DataModule(train_list,
                                   val_list,
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
   
    logger = TensorBoardLogger(save_dir=f"{config['model_dir']}/logs")
                                             
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        monitor="val_meanF1Score",
        dirpath= f"{config['model_dir']}/checkpoints/",
        filename="best_checkpoint_{epoch}_{val_meanF1Score:.2f}_{val_loss:.2f}",
        save_top_k=-1,  
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
        
    trainer.fit(seg_module, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)

def main(config_path:str):
        
    config =  read_config(config_path)

    arch_list = config["architecture"]
    encoder_list = config["encoder"]

    for arch in arch_list:
        config["architecture"] = str(arch)
        
        for encoder in encoder_list:  
            
            config["encoder"] = str(encoder)
            seed_everything(config["seed"], workers = True)
            
            
            config["devices"] =  int(os.environ["SLURM_GPUS_ON_NODE"])
            config["num_nodes"] =  int(os.environ["SLURM_NNODES"])
            config["batch_size"] = config["batch_size"] 
            
            print_config(config)
         
            timestamp = datetime.datetime.now().strftime('%Y%m%d/%H%M%S')
            model_name = f"{config['out_model_name']}_{config['architecture']}_{config['encoder']}_{config['num_epochs']}_es_{config['early_stopping_patience']}_bs_{config['batch_size']}"
            out_dir = Path(config["out_folder"], model_name, timestamp)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(config_path, out_dir)
            
            config["model_dir"] = out_dir
            
            config["out_model_name"] = model_name
           
            run_train(config)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python train.py <config_path>')
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
