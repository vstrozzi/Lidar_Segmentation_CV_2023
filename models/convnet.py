import torch
from torch import nn
from .basemodel import BaseModel

class ConvNet(BaseModel):
    def __init__(self, conv, loss, eval_metric, input_channels=3, output_layers=1, optimizer=torch.optim.Adam, lr=1e-4, prefix=None, suffix=None, use_clf=True):
        super().__init__(loss, eval_metric, optimizer, lr)

        self.conv = conv
        self.prefix = prefix
        self.suffix = suffix
        self.clf = nn.Linear(1000, output_layers)
        self.use_clf = use_clf        

    def forward(self, inputs):  
        out = inputs  
        if self.prefix is not None:    
            out = self.prefix(inputs)                            
        
        out = self.conv(out)        
        
        if self.use_clf:
            out = torch.flatten(out, start_dim=1)    
            out = self.clf(out)

        if self.suffix is not None:
            out = self.suffix(out)

        return torch.sigmoid(out)
