from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch

def SweepDataset(batch_size, transform):
    transform = transform
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                            transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = DataLoader(sub_dataset, batch_size=batch_size)

    return loader