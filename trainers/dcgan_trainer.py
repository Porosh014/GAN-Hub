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
from models.dcgan.generator import Generator
from models.dcgan.discriminator import Discriminator
from utils.utils import weights_init
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class DCGAN_Train:
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

        self.generator_network = Generator(self.config["noise_dim"], self.config["num_channels"],
                                           self.config["factor"]).to(self.device)

        self.discriminator_network = Discriminator(self.config["num_channels"], self.config["factor"]).to(self.device)

        # Initialize weights and biases
        # self.generator_network.apply(weights_init)
        # self.discriminator_network.apply(weights_init)

    def _initialize_trainers(self):
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(self.config['batch_size'], self.config["noise_dim"], 1, 1,
                                       device=self.device)

        self.real_label = 1.0
        self.fake_label = 0

        self.d_optimizer = optim.Adam(self.discriminator_network.parameters(),
                                      lr=self.config["learning_rate"], betas=(self.config["beta1"],
                                                                              self.config["beta2"]))
        self.g_optimizer = optim.Adam(self.generator_network.parameters(),
                                      lr=self.config["learning_rate"], betas=(self.config["beta1"],
                                                                              self.config["beta2"]))

    def _train_one_epoch(self):
        generator_loss = 0
        discriminator_loss = 0
        n_minibatch = 0
        self.discriminator_network.train()
        self.generator_network.train()

        for i, data in tqdm(enumerate(self.train_dataloader), desc="training-minibatch"):
            # reset the gradients
            self.discriminator_network.zero_grad()

            real_images = data.to(self.device)
            labels = torch.full((real_images.size(0),), self.real_label, device=self.device)

            output = self.discriminator_network(real_images).view(-1)
            real_error = self.criterion(output, labels)

            # generate fake images
            noise_batch = torch.randn(real_images.size(0), self.config["noise_dim"], 1, 1,
                                      device=self.device)
            fake_images = self.generator_network(noise_batch).to(self.device)
            labels = torch.full((real_images.size(0),), self.fake_label, device=self.device).float().cuda()

            # Forward pass the fake batch through discriminator
            output = self.discriminator_network(fake_images.detach()).view(-1)
            fake_error = self.criterion(output, labels)

            discriminator_error = (real_error + fake_error) / 2
            discriminator_error.backward()

            # Make a gradient step
            self.d_optimizer.step()

            ##############################
            ####  Generator training  ####
            ##############################

            self.generator_network.zero_grad()

            # create another batch of fakes for the generator
            noise_batch = torch.randn(real_images.size(0), self.config["noise_dim"], 1, 1,
                                      device=self.device)
            fake_images = self.generator_network(noise_batch).to(self.device)
            labels = torch.full((real_images.size(0),), self.real_label, device=self.device)

            # Forward pass the fake batch through generator
            output = self.discriminator_network(fake_images).view(-1)

            generator_error = self.criterion(output, labels)

            generator_error.backward()
            self.g_optimizer.step()

            generator_loss += generator_error.item()
            discriminator_loss += discriminator_error.item()
            n_minibatch += 1

        generator_loss = generator_loss / n_minibatch
        discriminator_loss = discriminator_loss / n_minibatch

        return generator_loss, discriminator_loss

    def _validate_one_epoch(self):
        generator_loss = 0
        discriminator_loss = 0
        n_minibatch = 0
        self.discriminator_network.eval()
        self.generator_network.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.val_dataloader), desc="validating:"):
                real_images = data.to(self.device)
                labels = torch.full((real_images.size(0),), self.real_label, device=self.device)

                output = self.discriminator_network(real_images).view(-1)
                real_error = self.criterion(output, labels)

                # generate fake images
                noise_batch = torch.randn(real_images.size(0), self.config["noise_dim"], 1, 1,
                                          device=self.device)
                fake_images = self.generator_network(noise_batch).to(self.device)
                labels = torch.full((real_images.size(0),), self.fake_label, device=self.device).float().cuda()

                # Forward pass the fake batch through discriminator
                output = self.discriminator_network(fake_images.detach()).view(-1)
                fake_error = self.criterion(output, labels)

                discriminator_error = (real_error + fake_error) / 2

                ##############################
                #### Generator validation ####
                ##############################

                # create another batch of fakes for the generator
                noise_batch = torch.randn(real_images.size(0), self.config["noise_dim"], 1, 1,
                                          device=self.device)
                fake_images = self.generator_network(noise_batch).to(self.device)
                labels = torch.full((real_images.size(0),), self.real_label, device=self.device)

                # Forward pass the fake batch through generator
                output = self.discriminator_network(fake_images).view(-1)

                generator_error = self.criterion(output, labels)

                generator_loss += generator_error.item()
                discriminator_loss += discriminator_error.item()
                n_minibatch += 1

            generator_loss = generator_loss / n_minibatch
            discriminator_loss = discriminator_loss / n_minibatch

            return generator_loss, discriminator_loss

    def _write_test_results(self):
        self.generator_network.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(self.test_dataloader), desc="writing output in disk:"):
                noise_batch = torch.randn(data.size(0), self.config["noise_dim"], 1, 1,
                                          device=self.device)

                fake_images = self.generator_network(noise_batch).to(self.device)

                save_image(fake_images, "output/images/%d.png" % i, nrow=8, normalize=True)

                if i == 100:
                    break

    def _train(self):

        for epoch in range(self.config["epochs"]):
            train_generator_loss, train_discriminator_loss = self._train_one_epoch()
            val_generator_loss, val_discriminator_loss = self._validate_one_epoch()

            self._write_test_results()

            print("epoch: {}, generator-train-loss: {}, discriminator-train-loss: {}, generator-val-loss: {},"
                  " discriminator-val-loss: {}".format(epoch + 1, train_generator_loss, train_discriminator_loss,
                                                       val_generator_loss, val_discriminator_loss))
