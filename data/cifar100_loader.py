# This script defines a function to load the CIFAR-100 dataset using PyTorch.
# It includes data augmentation and normalization transformations for training and testing sets.
# The function returns DataLoader objects for both training and testing datasets.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from collections import defaultdict
import random
import torch
from torchvision.datasets import CIFAR100
import numpy as np

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
    batch_size=50,
    seed=42,
    class_balanced=True,
    federatedTest=False
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
    train_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)

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

    for num_class in range(num_classes_per_client):
        for client_id in range(num_clients):
            i = 0
            while len(client_class_map[client_id]) < num_class:
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
    if federatedTest:
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

    # === Create dataloaders & datasets ===
    train_datasets = []
    for i in range(num_clients):
        train_subset = Subset(train_dataset, client_train_indices[i])
        train_datasets.append(train_subset)
    
    if federatedTest:
        test_loaders = []
        for i in range(num_clients):
            test_subset = Subset(test_dataset, client_test_indices[i])
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loaders.append(test_loader)
    else:
        test_loaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_datasets, test_loaders, client_class_map

def get_clustered_cifar100_datasets(
    n_clients_per_cluster=5,
    batch_size=50,
    seed=42,
    federatedTest=False
):
    random.seed(seed)
    np.random.seed(seed)
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

    # Load CIFAR-100 without transform initially
    full_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)

    # Use meta data from CIFAR-100 to get superclass → classes mapping
    meta = datasets.CIFAR100(root='./dataset', train=True, download=False, coarse=True)
    fine_labels = np.array(meta.targets)
    coarse_labels = np.array(meta.coarse_labels)

    # Build mapping from superclass → list of fine classes
    superclass_to_classes = defaultdict(set)
    for fine_label, coarse_label in zip(fine_labels, coarse_labels):
        superclass_to_classes[coarse_label].add(fine_label)

    # Group sample indices by fine label
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(fine_labels):
        class_to_indices[label].append(idx)

    # Build client datasets
    client_datasets = {}
    client_class_map = {}

    for superclass, fine_classes in superclass_to_classes.items():
        # Balance per class
        n_total_per_class = 500
        n_per_client = n_total_per_class // n_clients_per_cluster

        #DEBUG
        print(f"Super class {superclass} has fine classes: {fine_classes}")

        # Create clients
        for i in range(n_clients_per_cluster):
            indices = []
            for fine_label in fine_classes:
                selected = random.sample(class_to_indices[fine_label], n_per_client)
                indices.extend(selected)
                class_to_indices[fine_label] = list(set(class_to_indices[fine_label]) - set(selected))
            client_datasets[f"cluster{superclass}_client{i}"] = Subset(full_dataset, indices)
            client_class_map[f"cluster{superclass}_client{i}"] = fine_classes

    test_loaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return client_datasets, test_loaders, client_class_map
