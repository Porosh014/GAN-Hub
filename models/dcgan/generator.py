import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, n_channels, factor):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            # Input Size : 100 x 1 x 1
            nn.ConvTranspose2d(noise_dim, factor * 4, kernel_size=8, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            # Input Size : 128 x 8 x 8
            nn.ConvTranspose2d(factor * 4, factor * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # Input Size : 64 x 16 x 16
            nn.ConvTranspose2d(factor * 2, factor, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # Input Size : 32 x 32 x 32
            nn.ConvTranspose2d(factor, int(factor // 2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # Input Size : 16 x 64 x 64
            nn.ConvTranspose2d(int(factor // 2), n_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.generator(input)