import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, n_channels, factor):
        super(Generator, self).__init__()

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        # Input Size : 100 x 1 x 1
        self.trans_conv_0 = nn.ConvTranspose2d(noise_dim, factor * 4, kernel_size=8, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(factor * 4)

        self.trans_conv_1 = nn.ConvTranspose2d(factor * 4, factor * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(factor * 2)

        self.trans_conv_2 = nn.ConvTranspose2d(factor * 2, factor, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(factor)

        self.trans_conv_3 = nn.ConvTranspose2d(factor, int(factor // 2), kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(factor // 2))

        self.trans_conv_4 = nn.ConvTranspose2d(int(factor // 2), n_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        # Input Size : 100 x 1 x 1
        x = self.relu(self.bn0(self.trans_conv_0(x)))
        # Input Size : 128 x 8 x 8
        x = self.relu(self.bn1(self.trans_conv_1(x)))
        # Input Size : 64 x 16 x 16
        x = self.relu(self.bn2(self.trans_conv_2(x)))
        # Input Size : 32 x 32 x 32
        x = self.relu(self.bn3(self.trans_conv_3(x)))
        # Input Size : 16 x 64 x 64
        x = self.tanh(self.trans_conv_4(x))

        return x