import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channel, inner_channel, input_channel=None, sub_module=None, outer_most=False,
                 inner_most=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outer_channel = outer_channel
        self.inner_channel = inner_channel
        self.outer_most = outer_most
        if not input_channel:
            input_channel = outer_channel

        down_conv = nn.Conv2d(input_channel, inner_channel, kernel_size=4, stride=2, padding=1)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.InstanceNorm2d(inner_channel)

        up_relu = nn.ReLU(True)
        up_norm = nn.InstanceNorm2d(outer_channel)

        if outer_most:
            up_conv = nn.ConvTranspose2d(inner_channel * 2, outer_channel, kernel_size=4, stride=2, padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [sub_module] + up

        elif inner_most:
            up_conv = nn.ConvTranspose2d(inner_channel, outer_channel, kernel_size=4, stride=2, padding=1)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(inner_channel * 2, outer_channel, kernel_size=4, stride=2, padding=1)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]
            model = down + [sub_module] + up

        self.block = nn.Sequential(*model)

    def forward(self, x):
        # ic('forward')
        if self.outer_most:
            return self.block(x)
        else:  # add skip connections
            # ic(x.shape)
            # ic(self.outer_channel, self.inner_channel)
            return torch.cat([x, self.block(x)], 1)


class Unet(nn.Module):
    def __init__(self, input_channel, output_channel, num_downs):
        super(Unet, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(512, 512, sub_module=None, inner_most=True)  # add the innermost layer
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(512, 512, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(256, 512, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, sub_module=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, sub_module=unet_block)
        self.model = UnetSkipConnectionBlock(output_channel, 64, input_channel=input_channel, sub_module=unet_block,
                                             outer_most=True)  # add the outermost layer

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    tensor = torch.randn((5, 3, 256, 256))
    net = Unet(3, 3, 8)
    out = net(tensor)
    print(out.shape)


