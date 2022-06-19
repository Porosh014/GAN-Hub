import torch
from torch import nn
from models.block_modules.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, n_channel=3, block_iteration=3, dim_factor=2, init_dim=64, z_dim=128):
        super(Decoder, self).__init__()

        self.n_channel = n_channel
        self.block_iteration = block_iteration
        self.dim_factor = dim_factor
        self.init_dim = init_dim
        self.z_dim = z_dim
        self.max_dim = self.init_dim
        self.max_dim = 0

        for i in range(self.block_iteration):
            if i == 0:
                self.max_dim = self.init_dim

            else:
                self.max_dim *= self.dim_factor

        self.fc = nn.Sequential(nn.Linear(in_features=self.z_dim, out_features=8 * 8 * self.max_dim, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * self.max_dim, momentum=0.9),
                                nn.ReLU(True))

        decoder_layers = []

        for i in range(self.block_iteration):
            # not last block
            decoder_layers.append(DecoderBlock(channel_in=self.max_dim, channel_out=self.max_dim//self.dim_factor))
            self.max_dim //= self.dim_factor


        decoder_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.max_dim, out_channels=self.n_channel, kernel_size=5, stride=1,
                      padding=2),
            nn.Tanh()
        ))
        self.decoder_blocks = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(len(x), -1, 8, 8)
        x = self.decoder_blocks(x)

        return x