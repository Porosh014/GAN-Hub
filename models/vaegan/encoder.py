import torch
from torch import nn
from models.block_modules.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, n_channel=3, block_iteration=3, dim_factor=2, init_dim=64, z_dim=128):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.block_iteration = block_iteration
        self.dim_factor = dim_factor
        self.init_dim = init_dim
        self.z_dim = z_dim
        self.dim = self.n_channel

        encoder_layers = []

        for i in range(self.block_iteration):
            if i == 0:
                encoder_layers.append(EncoderBlock(channel_in=self.dim, channel_out=self.init_dim))
                self.dim = self.init_dim

            else:
                encoder_layers.append(EncoderBlock(channel_in=self.dim, channel_out=self.dim * self.dim_factor))
                self.dim *= self.dim_factor

        self.encoder_blocks = nn.Sequential(*encoder_layers)

        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.dim, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))

        self.layer_mu = nn.Linear(in_features=1024, out_features=self.z_dim)
        self.layer_std = nn.Linear(in_features=1024, out_features=self.z_dim)

    def forward(self, x):
        x = self.encoder_blocks(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        mu = self.layer_mu(x)
        logstd = self.layer_std(x)

        return mu, logstd



