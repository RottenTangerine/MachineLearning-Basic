import torch.nn as nn

latent_dim = 2 ** 7
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(_in_channels, _out_channels, normalize = True):
            layers = [nn.Conv2d(_in_channels, _out_channels)]
            if normalize:
                layers.append(nn.BatchNorm2d(_out_channels))
            layers.append(nn.ReLU(True))
            return layers

        self.deconvolution = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
        )
        self.linear = nn.Linear(1024, 28 * 28)

    def forward(self, x):
        out = self.deconvolution(x)
        out = self.linear(out)
        out = out.view(out.size(0), *img_shape)
        return out


