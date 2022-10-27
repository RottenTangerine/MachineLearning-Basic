import torch
import torch.nn as nn
from Basic_Conv import BasicConv
from Basic_ResidualBlock import BasicResidualBlock


class EncodeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncodeBlock, self).__init__()
        self.pooling = nn.AvgPool2d((2, 2), 2)  # avoid CUDA out of memory
        self.conv = BasicConv(input_channel, output_channel)
        self.residual = BasicResidualBlock(output_channel, output_channel)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return self.residual(x)


class DecodeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecodeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv(input_channel, output_channel)   # avoid CUDA out of memory
        self.conv_past = BasicConv(input_channel, output_channel)
        self.res = BasicResidualBlock(output_channel * 2, output_channel)
        self.conv2 = BasicConv(output_channel, output_channel)

    def forward(self, x, p):
        x = self.conv(x)
        p = self.conv_past(p)
        x = self.res(torch.cat([x, p], dim=1))
        x = self.upsample(x)
        return self.conv2(x)


# input size 256x256x3
class UNet(nn.Module):
    def __init__(self, args, n_residual_block=9):
        super(UNet, self).__init__()

        # initial conv
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(args.in_channel, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        # 256x256x64

        # down sampling
        self.conv1 = EncodeBlock(64, 128)
        self.conv2 = EncodeBlock(128, 256)

        # residual layers
        layers = []
        for _ in range(n_residual_block):
            layers.append(BasicResidualBlock(256, 256))
        self.residual_layers = nn.Sequential(*layers)

        # up sampling
        self.dconv2 = DecodeBlock(256, 128)
        self.dconv1 = DecodeBlock(128, 64)

        # output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, args.out_channel, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init_conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.residual_layers(x2)
        x = self.dconv2(x, x2)
        x = self.dconv1(x, x1)
        x = self.output_layer(x)
        return x
