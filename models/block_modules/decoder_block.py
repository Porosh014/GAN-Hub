import torch
import torch.nn.functional as F
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, True)
        return x
