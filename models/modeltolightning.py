import torch
from torch import nn
import pytorch_lightning as pl
from utils.dataloader.labels import *
import wandb

class LightningModel(pl.LightningModule):
    def __init__(self, model, loss, eval_metric, optimizer=torch.optim.Adam, lr=1e-4, mode="RGB", logger_mode="None"):
        super().__init__()

        self.model = model
        self.dict = {x.color:((x.trainId)) for x in labels}
        self.loss = loss
        self.eval_metric = eval_metric
        self.optimizer = optimizer
        self.lr = lr
        self.mode = mode
        self.logger_mode = logger_mode

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, train_batch, batch_idx):
        # Test model on one sample

        x, y = train_batch

        if self.mode == "LIDAR":
            out = x["left_disp"]
        elif self.mode == "LIDAR-RGB":
            out = 0.95*x["left_rgb"] + 0.05*torch.cat((x["left_disp"], x["left_disp"], x["left_disp"]), 1)
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
            out = 0.95*x["left_rgb"] + 0.05*torch.cat((x["left_disp"], x["left_disp"], x["left_disp"]), 1)
        else:
            out = x["left_rgb"]

        
        out = self.forward(out)
        loss = self.loss(out, y)
        
        pred = torch.argmax(out, 1)
        score = self.eval_metric(pred.squeeze(), y.squeeze())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)

        # Only on first batch
        if self.logger_mode == "WANDB":
            if batch_idx == 0:
                mask_list = []
                for i in range(0, y.shape[0]):
                    class_labels = {0: "person", 1: "vehicle", 2: "rider", 3:"others"}  
                    mask_img = wandb.Image(
                        x["left_rgb"][i],
                        masks={
                            "predictions": {"mask_data": pred[i].numpy(force=True), "class_labels": class_labels},
                            "ground_truth": {"mask_data": y[i].squeeze().numpy(force=True), "class_labels": class_labels},
                        })
                    mask_list.append(mask_img)

                self.logger.experiment.log({"image": mask_list})
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
