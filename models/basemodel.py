import torch
from torch import nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, loss, eval_metric, optimizer=torch.optim.Adam, lr=1e-4):
        super().__init__()

        self.loss = loss
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, inputs):
        return inputs
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.loss(out.squeeze(), y.squeeze())        
        self.log('train_loss', loss)        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss(out.squeeze(), y.squeeze())
        score = self.eval_metric(out.squeeze().cpu().round(), y.squeeze().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)