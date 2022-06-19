import json
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision.transforms as transforms

from data_utils.create_split import create_train_val_split
from data_utils.celeba_loader import CelebA

from trainers.dcgan_trainer import DCGAN_Train
from trainers.vaegan_trainer import VAEGAN_Train

class Train:

    def __init__(self, opt):
        self.model_name = opt.model_name
        self.dataset_root = opt.dataset_root
        self.config_path = opt.config_path

        with open(self.config_path) as file:
            self.config = json.load(file)

        self.transform = transforms.Compose([
            transforms.Resize((self.config["input_size"], self.config["input_size"])),
            transforms.CenterCrop(self.config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])

        train_splits, val_splits, test_splits = create_train_val_split(self.dataset_root)

        self.train_data = CelebA(self.dataset_root, train_splits, self.transform)
        self.val_data = CelebA(self.dataset_root, val_splits, self.transform)
        self.test_data = CelebA(self.dataset_root, test_splits, self.transform)

        self.dataset = [self.train_data, self.val_data, self.test_data]

        # self.trainer = DCGAN_Train(self.dataset, self.config)
        # self.trainer._initialize_trainers()
        # self.trainer._train()

        self.trainer = VAEGAN_Train(self.dataset, self.config)
        self.trainer._initialize_trainers()
        self.trainer._train_one_epoch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dcgan', help='project name specified by backbone')
    parser.add_argument('--dataset_root', type=str, default='dataset/archive', help='dataset name')
    parser.add_argument('--config_path', type=str, default='config/vaegan_config.json', help='config file path')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='weight file path')
    opt = parser.parse_args()

    train = Train(opt)
    train._initialize_trainers()
    train._train()

