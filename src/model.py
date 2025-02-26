import torch
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import MetricCollection, MeanMetric, JaccardIndex, Accuracy, F1Score
from monai.losses import DiceCELoss

from utils import load_model

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
            self.class_weights = torch.FloatTensor(np.array(list(config['weights_classes'].values()))[:,0])
        self.criterion = DiceCELoss(softmax=True,
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
        
    
    
    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        
        output = self.train_metrics.compute()
        self.log_dict({"train_loss": train_loss.item(),
                           
                   "train_mean_IoU": torch.mean(output["train_Class_IoU"]),
                   "train_meanF1Score": torch.mean(output["train_F1Score"]),
                   "train_meanAccuracy": torch.mean(output["train_Accuracy"]),
                   
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

                       "val_mean_IoU": torch.mean(output["val_Class_IoU"]),
                       "val_meanF1Score": torch.mean(output["val_F1Score"]),
                       "val_meanAccuracy": torch.mean(output["val_Accuracy"]),
                       
                       
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