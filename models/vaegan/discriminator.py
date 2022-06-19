import torch
from torch import nn
import torch.nn.functional as F
from models.block_modules.encoder_block import EncoderBlock

class Discriminator(nn.Module):
    def __init__(self, n_channel=3, block_iteration=3, dim_factor=2, init_dim=64, z_dim=128):
        super(Discriminator, self).__init__()

        self.n_channel = n_channel
        self.block_iteration = block_iteration
        self.dim_factor = dim_factor
        self.init_dim = init_dim
        self.z_dim = z_dim
        self.dim = self.n_channel

        discriminator_layers = []

        for i in range(self.block_iteration):

            if i == 0:
                discriminator_layers.append(EncoderBlock(channel_in=self.dim, channel_out=self.init_dim))
                self.dim = self.init_dim

            else:
                discriminator_layers.append(EncoderBlock(channel_in=self.dim, channel_out=self.dim * self.dim_factor))
                self.dim *= self.dim_factor

        self.discriminator_blocks = nn.Sequential(*discriminator_layers)

        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.dim, out_features=512, bias=False),
                                nn.BatchNorm1d(num_features=512, momentum=0.9),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=1),
                                )

    def forward(self, x, x_predicted, x_sampled, mode='reconstruction'):
        x_all = torch.cat((x, x_predicted, x_sampled), 0)

        if mode == "reconstruction":

            for i, block in enumerate(self.discriminator_blocks):

                if i == (self.block_iteration - 1):
                    x_all, block_x = block(x_all, True)
                    block_x = block_x.view(len(block_x), -1)

                    return block_x

                else:
                    x_all = block(x_all)

        else:

            for i, block in enumerate(self.discriminator_blocks):
                x_all = block(x_all)

            x_all = x_all.view(len(x_all), -1)
            x_all = self.fc(x_all)

            return F.sigmoid(x_all)