import torch.nn as nn

img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(_in_channels, _out_channels, normalize=True):
            layers = [nn.Conv2d(_in_channels, _out_channels, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(_out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.deconvolution = nn.Sequential(
            *block(1, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )


    def forward(self, x):
        out = self.deconvolution(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = out.view(out.size(0), *img_shape)
        return out


if __name__ == '__main__':
    import torch, numpy as np
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    z = torch.FloatTensor(np.random.normal(0, 1, (10, 2 ** 8))).to(device)
    z = z.view(z.size(0), 1, 2**4, 2**4)

    generator = Generator().to(device)
    out = generator(z)
    print(out)