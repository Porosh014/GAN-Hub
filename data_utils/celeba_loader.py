import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CelebA(Dataset):
    def __init__(self, dataset_root, split_filenames, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.dataset_path = os.path.join(self.dataset_root, 'img_align_celeba', 'img_align_celeba')
        self.image_files = sorted(split_filenames)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.dataset_path,
                                self.image_files[idx])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image