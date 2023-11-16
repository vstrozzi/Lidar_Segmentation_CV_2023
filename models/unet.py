import torch
from torch import nn
from .blocks.blocks import ConvBlock, AttentionBlock, EncoderBlock, DecoderBlock
from .basemodel import BaseModel

class UNet(BaseModel):
    def __init__(self, loss, eval_metric, channels, output_layers=1, f=64, img_shape=(256, 256), activate_clf=True, att=False, optimizer=torch.optim.Adam, lr=1e-4, postprocess=None):
        super().__init__(loss, eval_metric, optimizer, lr)

        self.e1 = EncoderBlock(channels, f)
        self.e2 = EncoderBlock(f, 2*f)
        self.e3 = EncoderBlock(2*f, 4*f)
        self.e4 = EncoderBlock(4*f, 8*f)

        self.b = ConvBlock(8*f, 16*f)

        self.d1 = DecoderBlock(16*f, 8*f, AttentionBlock(8*f) if att else None)
        self.d2 = DecoderBlock(8*f, 4*f, AttentionBlock(4*f) if att else None)
        self.d3 = DecoderBlock(4*f, 2*f, AttentionBlock(2*f) if att else None)
        self.d4 = DecoderBlock(2*f, f, AttentionBlock(1*f) if att else None)

        self.output = nn.Conv2d(f, 1, kernel_size=1, padding=0)
        #self.output2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.clf = nn.Linear(img_shape[0]*img_shape[1], output_layers)
        self.activate_clf = activate_clf

        self.postprocess = postprocess

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        out = self.output(d4)
        #out = self.output2(out)

        if self.activate_clf:
            out = torch.flatten(out, start_dim=1)
            out = self.clf(out)        

        return torch.sigmoid(out)
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss(out.squeeze(), y.squeeze())
        out = out.squeeze().cpu().round()
        if self.postprocess != None:
            out = self.postprocess(out)
        score = self.eval_metric(out, y.squeeze().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
