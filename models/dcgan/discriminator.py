import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_channels, factor):
        super(Discriminator, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.conv0 = nn.Conv2d(n_channels, factor, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(factor)

        self.conv1 = nn.Conv2d(factor, factor * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(factor * 2)

        self.conv2 = nn.Conv2d(factor * 2, factor * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(factor * 4)

        self.conv3 = nn.Conv2d(factor * 4, factor * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(factor * 2)

        self.conv4 = nn.Conv2d(factor * 2, 1, kernel_size=8, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.leaky_relu(self.bn0(self.conv0(x)))
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))

        return x
