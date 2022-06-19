import time
import torch
import json
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.autograd import Variable
from utils.utils import weights_init
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.vaegan.vaegan import VAEGAN

torch.autograd.set_detect_anomaly(True)

class VAEGAN_Train:
    def __init__(self, dataset, config):

        self.dataset = dataset
        self.config = config

        self.train_dataloader = DataLoader(self.dataset[0], batch_size=self.config['batch_size'], shuffle=True,
                                           num_workers=4)
        self.val_dataloader = DataLoader(self.dataset[1], batch_size=self.config['batch_size'], shuffle=False,
                                         num_workers=4)
        self.test_dataloader = DataLoader(self.dataset[2], batch_size=self.config['batch_size'], shuffle=False,
                                          num_workers=4)

        self.device = torch.device("cuda")

        self.network = VAEGAN(self.config).to(self.device)


    def _initialize_trainers(self):

        self.optimizer_encoder = RMSprop(params=self.network.encoder.parameters(), lr=self.config["learning_rate"],
                                         alpha=0.9, eps=1e-8, weight_decay=0,
                                         momentum=0, centered=False)
        self.lr_encoder = ExponentialLR(self.optimizer_encoder, gamma=self.config["decay_lr"])

        self.optimizer_decoder = RMSprop(params=self.network.decoder.parameters(), lr=self.config["learning_rate"],
                                         alpha=0.9, eps=1e-8, weight_decay=0,
                                         momentum=0, centered=False)
        self.lr_decoder = ExponentialLR(self.optimizer_decoder, gamma=self.config["decay_lr"])

        self.optimizer_discriminator = RMSprop(params=self.network.discriminator.parameters(),
                                               lr=self.config["learning_rate"],
                                               alpha=0.9, eps=1e-8,
                                               weight_decay=0, momentum=0, centered=False)
        self.lr_discriminator = ExponentialLR(self.optimizer_discriminator, gamma=self.config["decay_lr"])

    def _train_one_epoch(self):

        for i, data in tqdm(enumerate(self.train_dataloader), desc="training-minibatch"):

            self.network.train()

            data = Variable(data, requires_grad=False).float().cuda()
            x_t, disc_class, disc_layer, mus, log_variances = self.network(data)

            # split so we can get the different parts
            disc_layer_original = disc_layer[:self.config["batch_size"]]
            disc_layer_predicted = disc_layer[self.config["batch_size"]:-self.config["batch_size"]]
            disc_layer_sampled = disc_layer[-self.config["batch_size"]:]

            disc_class_original = disc_class[:self.config["batch_size"]]
            disc_class_predicted = disc_class[self.config["batch_size"]:-self.config["batch_size"]]
            disc_class_sampled = disc_class[-self.config["batch_size"]:]

            nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = VAEGAN.loss(data, x_t, disc_layer_original,
                                                                                             disc_layer_predicted,
                                                                                             disc_layer_sampled,
                                                                                             disc_class_original,
                                                                                             disc_class_predicted,
                                                                                             disc_class_sampled, mus,
                                                                                             log_variances)


            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            loss_encoder = torch.sum(kl) + torch.sum(mse)
            loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
            loss_decoder = torch.sum(self.config["lambda_mse"] * mse) - (1.0 - self.config["lambda_mse"]) * loss_discriminator

            self.network.zero_grad()

            # encoder
            # someone likes to clamp the grad here:[p.grad.data.clamp_(-1, 1) for p in self.network.encoder.parameters()]

            loss_encoder.backward(retain_graph=True)
            loss_decoder.backward(retain_graph=True)
            loss_discriminator.backward()

            #decoder [p.grad.data.clamp_(-1, 1) for p in self.network.decoder.parameters()]
            self.optimizer_encoder.step()
            self.optimizer_decoder.step()
            self.optimizer_discriminator.step()

            self.optimizer_encoder.zero_grad()
            self.optimizer_decoder.zero_grad()
            self.optimizer_discriminator.zero_grad()
            # #discriminator
            # loss_discriminator.backward()  # [p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
            # self.optimizer_discriminator.step()

            print('[%02d] encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (
            i, loss_encoder, loss_decoder, loss_discriminator))
