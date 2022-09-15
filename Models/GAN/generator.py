import torch.nn as nn
import numpy as np

latent_dim = 2 ** 8
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(_in_channels, _out_channels, normalize=True):
            layers = [nn.Linear(_in_channels, _out_channels)]
            if normalize:
                layers.append(nn.BatchNorm1d(_out_channels, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.deconvolution = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )


    def forward(self, x):
        out = self.deconvolution(x)
        out = self.linear(out)
        out = out.view(out.size(0), *img_shape)
        return out


if __name__ == '__main__':
    import torch, numpy as np
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    z = torch.FloatTensor(np.random.normal(0, 1, (5, latent_dim))).to(device)
    print(z)

    generator = Generator().to(device)
    out = generator(z)
    print(out)