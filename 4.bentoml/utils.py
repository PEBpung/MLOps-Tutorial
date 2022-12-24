import os
import torch
import random
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms


def get_dataset():
    train_set = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    test_set = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
    return train_set, test_set


def seed_everything(seed=42):
    # reproducible setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.sum += val
        self.count += count

    def get_avg(self):
        return self.sum / self.count if self.count > 0 else 1e-4
