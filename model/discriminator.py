from torch import nn

class PatchGANDiscriminator(nn.Module):

    def __init__(self, input_nc, n_filters=64, n_layers=3):
        super().__init__()
        sequence = [nn.Conv2d(input_nc, n_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(n_filters * nf_mult_prev, n_filters * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(n_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.seq = nn.Sequential(*sequence)

    def forward(self, input):
        return self.seq(input)
