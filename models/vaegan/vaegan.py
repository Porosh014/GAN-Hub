import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models.vaegan.encoder import Encoder
from models.vaegan.decoder import Decoder
from models.vaegan.discriminator import Discriminator

class VAEGAN(nn.Module):
    def __init__(self, config):
        super(VAEGAN, self).__init__()

        self.config = config

        self.encoder = Encoder(self.config["n_channels"], self.config["block_iteration"],
                               self.config["dim_factor"], self.config["init_dim"], self.config["z_dim"])

        self.decoder = Decoder(self.config["n_channels"], self.config["block_iteration"],
                               self.config["dim_factor"], self.config["init_dim"], self.config["z_dim"])

        self.discriminator = Discriminator(self.config["n_channels"], self.config["block_iteration"],
                                           self.config["dim_factor"], self.config["init_dim"], self.config["z_dim"])

        # self.init_parameters()

    # def init_parameters(self):
    #     # just explore the network, find every weight and bias matrix and fill it
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
    #             if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
    #                 # init as original implementation
    #                 scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
    #                 scale /= np.sqrt(3)
    #                 # nn.init.xavier_normal(m.weight,1)
    #                 # nn.init.constant(m.weight,0.005)
    #                 nn.init.uniform(m.weight, -scale, scale)
    #             if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant(m.bias, 0.0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)


    def forward(self, x):

        mu, logstd = self.encoder(x)
        z = self.reparameterize(mu.clone(), logstd.clone())
        x_t = self.decoder(z)

        z_p = Variable(torch.randn(len(x), self.config["z_dim"]).cuda(), requires_grad=True)
        x_p = self.decoder(z_p)

        disc_layer = self.discriminator(x, x_t, x_p, "reconstruction")  # discriminator for reconstruction
        disc_class = self.discriminator(x, x_t, x_p, "gan")

        return x_t, disc_class, disc_layer, mu, logstd

    @staticmethod
    def loss(x, x_t, disc_layer_original, disc_layer_predicted, disc_layer_sampled, disc_class_original,
             disc_class_predicted, disc_class_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x.view(len(x), -1) - x_t.view(len(x_t), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        # mse between intermediate layers
        mse = torch.sum(0.5 * (disc_layer_original - disc_layer_predicted) ** 2, 1)

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(disc_class_original + 1e-3)
        bce_dis_predicted = -torch.log(1 - disc_class_predicted + 1e-3)
        bce_dis_sampled = -torch.log(1 - disc_class_sampled + 1e-3)

        return nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled