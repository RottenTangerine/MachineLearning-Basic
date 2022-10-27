import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, bn=True, activate=True):
        super(BasicConv, self).__init__()
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride)
        ]
        if bn:
            layers.append(nn.InstanceNorm2d(output_channel))
        if activate:
            layers.append(nn.LeakyReLU(0.2, True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)