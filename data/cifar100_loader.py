# This script defines a function to load the CIFAR-100 dataset using PyTorch.
# It includes data augmentation and normalization transformations for training and testing sets.
# The function returns DataLoader objects for both training and testing datasets.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from collections import defaultdict
import random
import torch

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)

def get_cifar100_loaders(batch_size=64, val_split=0.1):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])

    # Load dataset without transforms
    full_train_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=None)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Wrap subsets with appropriate transforms
    train_dataset = TransformedDataset(train_subset, transform_train)
    val_dataset = TransformedDataset(val_subset, transform_test)

    test_dataset = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# Federated CIFAR-100 Dataloaders
# This function creates federated dataloaders for CIFAR-100 dataset.
def get_federated_cifar100_dataloaders(
    num_clients,
    num_classes_per_client,
    batch_size=64,
    seed=42,
    class_balanced=True
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Load CIFAR-100
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    def group_by_class(dataset):
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_to_indices[label].append(idx)
        return class_to_indices

    train_class_indices = group_by_class(train_dataset)
    test_class_indices = group_by_class(test_dataset)

    # === Assign classes to clients ===
    max_clients_per_class = (num_clients * num_classes_per_client) // 100
    class_pool = list(range(100)) * max_clients_per_class
    random.shuffle(class_pool)

    class_client_map = defaultdict(list)
    client_class_map = [set() for _ in range(num_clients)]

    for client_id in range(num_clients):
        i = 0
        while len(client_class_map[client_id]) < num_classes_per_client:
            if i >= len(class_pool):
                raise RuntimeError("Insufficient class slots to assign classes to all clients.")
            cls = class_pool[i]
            if client_id not in class_client_map[cls]:
                client_class_map[client_id].add(cls)
                class_client_map[cls].append(client_id)
                class_pool.pop(i)
            else:
                i += 1

    # === Allocate training data ===
    used_train_indices = set()
    client_train_indices = [[] for _ in range(num_clients)]

    total_samples = 50000
    samples_per_client = total_samples // num_clients

    for client_id, class_set in enumerate(client_class_map):
        if class_balanced:
            per_class_quota = samples_per_client // num_classes_per_client
            for cls in class_set:
                available = [i for i in train_class_indices[cls] if i not in used_train_indices]
                if len(available) < per_class_quota:
                    raise ValueError(f"Not enough samples in class {cls} for client {client_id}")
                selected = available[:per_class_quota]
                used_train_indices.update(selected)
                client_train_indices[client_id].extend(selected)
        else:
            all_available = []
            for cls in class_set:
                all_available += [i for i in train_class_indices[cls] if i not in used_train_indices]
            if len(all_available) < samples_per_client:
                raise ValueError(f"Not enough total samples for client {client_id}")
            random.shuffle(all_available)
            selected = all_available[:samples_per_client]
            used_train_indices.update(selected)
            client_train_indices[client_id].extend(selected)

    # === Allocate test data ===
    used_test_indices = set()
    client_test_indices = [[] for _ in range(num_clients)]

    for cls, clients_with_class in class_client_map.items():
        available = [i for i in test_class_indices[cls] if i not in used_test_indices]
        n = len(available)
        if n < len(clients_with_class):
            raise ValueError(f"Not enough test samples for class {cls}")
        chunk_size = n // len(clients_with_class)
        remainder = n % len(clients_with_class)
        start = 0
        for i, client_id in enumerate(clients_with_class):
            extra = 1 if i < remainder else 0
            end = start + chunk_size + extra
            selected = available[start:end]
            used_test_indices.update(selected)
            client_test_indices[client_id].extend(selected)
            start = end

    # === Create dataloaders ===
    train_loaders = []
    test_loaders = []
    for i in range(num_clients):
        train_subset = Subset(train_dataset, client_train_indices[i])
        test_subset = Subset(test_dataset, client_test_indices[i])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, client_class_map
