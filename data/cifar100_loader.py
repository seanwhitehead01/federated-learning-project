import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torch.serialization import safe_globals

def get_cifar100_dataloaders(
    dataset_dir="dataset",
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle_train=True,
):
    train_path = f"{dataset_dir}/cifar100_train.pt"
    test_path = f"{dataset_dir}/cifar100_test.pt"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Dataset not found. Run download_cifar100.py first.")

    with safe_globals([CIFAR100]):
        train_set = torch.load(train_path, weights_only=False)
        test_set = torch.load(test_path, weights_only=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
