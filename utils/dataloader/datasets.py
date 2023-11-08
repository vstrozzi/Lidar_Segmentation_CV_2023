import cv2
import torch
from torch.utils.data import Dataset
from .transforms import transform_default


class RSDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
        
    def __len__(self):
        return self.data.shape[0]


class AllInOneDataset(Dataset):
    def __init__(self, data, labels, patch_size=(16, 16), data_transform=transform_default, label_transform=transform_default, extra_layers_transf=[], single_label=False):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.extra_layers_transf = extra_layers_transf
        self.single_label = single_label

    def __getitem__(self, index):
        data, i1, i2 = self.data[index]
        data = cv2.imread(data)
        raw_label = cv2.imread(self.labels[index])        
        data = self.data_transform(data)[:,i1*self.patch_size[0]:(i1+1)*self.patch_size[0], i2*self.patch_size[1]:(i2+1)*self.patch_size[1]]
        label = self.label_transform(raw_label).squeeze()          
        
        if self.single_label:
            label = label[i1*self.patch_size[0]:(i1+1)*self.patch_size[0], i2*self.patch_size[1]:(i2+1)*self.patch_size[1]]
        
        for transf in self.extra_layers_transf:            
            data = torch.cat((data, transf(raw_label)), dim=0)
        
        return data, label
        
    def __len__(self):
        return len(self.data)