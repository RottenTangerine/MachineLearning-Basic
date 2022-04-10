import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out:= out_channels * ResidualBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out)
            )


    def forward(self, x):
        return nn.ReLU(True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.in_channels = 64

        self.conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # one layer may contain more than one residual block
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels =  out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(1), -1)

        return self.fc(x)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.randn(20, 3, 400, 400).to(device)
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
    out = model(x)

