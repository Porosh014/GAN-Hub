import torch
import torch.nn.functional as F
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, x, out=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            x = self.conv(x)
            x_out = x
            x = self.bn(x)
            x = F.relu(x, False)
            return x, x_out
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x, True)
            return x
