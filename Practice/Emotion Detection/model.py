import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, class_num):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, True),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(512, class_num, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        return out
