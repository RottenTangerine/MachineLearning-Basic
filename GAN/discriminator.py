import torch
import torch.nn as nn

# img shape (1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        small_cfg = [32, 32, 'M', 64, 64, 'M', 128, 'M']
        in_channels = 1

        layers = []
        for v in small_cfg:
            if v == 'M': layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, 3, padding=1)]
                in_channels = v

        self.feature = nn.Sequential(*layers)
        self.avgPool = nn.AdaptiveAvgPool2d(5)
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 2 ** 11),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2 ** 11, 2 ** 11),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2 ** 11, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.feature(x)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    discriminator = Discriminator().to(device)
    sample_img = torch.rand((5, 1, 28, 28)).to(device)
    out = discriminator(sample_img)
    print(out)