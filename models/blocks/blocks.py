import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.wx = nn.Conv2d(c, c, 1)
        self.wg = nn.Conv2d(c, c, 1)
        self.psi = nn.Conv2d(c, c, 1)
    def forward(self, x, g):
        x_0 = x
        x = self.wx(x)
        g = self.wg(g)
        x = nn.functional.relu(x + g)
        x = self.psi(x)
        x = torch.sigmoid(x)
        return x_0 * (x + 1)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, att_block=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)
        self.att_block = att_block

    def forward(self, inputs, skip):
        x = self.up(inputs)
        if self.att_block is not None:
            skip = self.att_block(skip, x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
