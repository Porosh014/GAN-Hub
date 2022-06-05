import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_channels, factor):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # Input Size: 3 x 128 x 128
            nn.Conv2d(n_channels, factor, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 32 x 64 x 64
            nn.Conv2d(factor, factor * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 64 x 32 x 32
            nn.Conv2d(factor * 2, factor * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 128 x 16 x 16
            nn.Conv2d(factor * 4, factor * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 64 x 8 x 8
            nn.Conv2d(factor * 2, 1, kernel_size=8, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            # Output Size : 1 x 1 x 1
        )

    def forward(self, input):

        return self.discriminator(input)