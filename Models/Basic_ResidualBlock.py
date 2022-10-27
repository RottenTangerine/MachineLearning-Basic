import torch
import torch.nn as nn
from Basic_Conv import BasicConv


class BasicResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = BasicConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

