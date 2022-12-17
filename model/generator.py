import torch
from torch import nn


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, n_layers, n_filters=64, use_dropout=False):
        super().__init__()
        sc_block = SkipConnection(n_filters * 8, n_filters * 8, input_nc=None, submodule=None, inner_block=True)
        for _ in range(n_layers - 5):
            sc_block = SkipConnection(n_filters * 8, n_filters * 8, input_nc=None, submodule=sc_block, use_dropout=use_dropout)
        sc_block = SkipConnection(n_filters * 4, n_filters * 8, input_nc=None, submodule=sc_block)
        sc_block = SkipConnection(n_filters * 2, n_filters * 4, input_nc=None, submodule=sc_block)
        sc_block = SkipConnection(n_filters, n_filters * 2, input_nc=None, submodule=sc_block)
        self.outer_block = SkipConnection(output_nc, n_filters, input_nc=input_nc, submodule=sc_block, outer_block=True)

    def forward(self, x):
        return self.outer_block(x)


class SkipConnection(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outer_block=False, inner_block=False, use_dropout=False):
        super().__init__()
        self.outer_block = outer_block
        if input_nc is None:
            input_nc = outer_nc
        conv_down = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        relu_down = nn.LeakyReLU(0.2, True)
        norm_down = nn.BatchNorm2d(inner_nc)
        relu_up = nn.ReLU(True)
        norm_up = nn.BatchNorm2d(outer_nc)

        if outer_block:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [conv_down]
            up = [relu_up, upconv, nn.Tanh()]
            sequence = down + [submodule] + up
        elif inner_block:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [relu_down, conv_down]
            up = [relu_up, upconv, norm_up]
            sequence = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [relu_down, conv_down, norm_down]
            up = [relu_up, upconv, norm_up]

            if use_dropout:
                sequence = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                sequence = down + [submodule] + up

        self.seq = nn.Sequential(*sequence)

    def forward(self, x):
        if self.outer_block:
            return self.seq(x)
        else:
            return torch.cat([x, self.seq(x)], dim=1)
