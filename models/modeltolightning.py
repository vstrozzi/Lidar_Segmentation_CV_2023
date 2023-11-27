import torch
from torch import nn
import pytorch_lightning as pl
from utils.dataloader.labels import *

class LightningModel(pl.LightningModule):
    def __init__(self, model, loss, eval_metric, optimizer=torch.optim.Adam, lr=1e-4, mode="RGB"):
        super().__init__()

        self.model = model
        self.dict = {x.color:((x.trainId)) for x in labels}
        self.loss = loss
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.lr = lr
        self.mode = mode

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, train_batch, batch_idx):
        # Test model on one sample

        x, y = train_batch

        if self.mode == "LIDAR":
            out = x["left_disp"]
        elif self.mode == "LIDAR-RGB":
            out = torch.cat((x["left_rgb"], x["left_disp"]), 1)
        else:
            out = x["left_rgb"]

        out = self.forward(out)
        loss = self.loss(out, y)   

        pred = torch.argmax(out, 1)
        score = self.eval_metric(pred.squeeze(), y.squeeze())
        self.log('train_loss', loss)      
        self.log('train_score', score)  
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        if self.mode == "LIDAR":
            out = x["left_disp"]
        elif self.mode == "LIDAR-RGB":
            out = torch.cat((x["left_rgb"], x["left_disp"]), 1)
        else:
            out = x["left_rgb"]

        out = self.forward(out)
        loss = self.loss(out, y)
        
        pred = torch.argmax(out, 1)
        score = self.eval_metric(pred.squeeze().cpu().round(), y.squeeze().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)