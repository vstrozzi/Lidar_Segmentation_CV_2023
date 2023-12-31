import torch
from torch import nn
import pytorch_lightning as pl
from utils.dataloader.labels import *

class BaseModel(pl.LightningModule):
    def __init__(self, loss, eval_metric, optimizer=torch.optim.Adam, lr=1e-4):
        super().__init__()

        self.dict = {x.color:((x.trainId)) for x in labels}
        self.loss = loss
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, inputs):
        return inputs
    
    def training_step(self, train_batch, batch_idx):
        # Test model on one sample

        x, y = train_batch
        y = torch.tensor(list(map(lambda k: RGBtoOneHot(k, dict).astype(int), y)))

        out = self.forward(x["left_rgb"])
        loss = self.loss(out.squeeze(), y.squeeze())   
        pred = torch.argmax(out, 1)
        score = self.eval_metric(pred.squeeze(), y.squeeze())
        self.log('train_loss', loss)      
        self.log('train_score', score)  
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = torch.tensor(list(map(lambda k: RGBtoOneHot(k, dict).astype(int), y)))
        out = self.forward(x["left_rgb"])
        loss = self.loss(out.squeeze(), y.squeeze())
        pred = torch.argmax(out, 1)
        score = self.eval_metric(pred.squeeze().cpu().round(), y.squeeze().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)