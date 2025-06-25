from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from collections import defaultdict
import random
import torch
import os
import pickle
import numpy as np

class TransformedDataset(Dataset):
    """A wrapper dataset that applies transformations to the CIFAR-100 dataset."""

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
    """Load CIFAR-100 dataset for centralized scenario.
    Args:
        batch_size: Batch size for DataLoader.
        val_split: Fraction of training data to use for validation.
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
    """
    # Define transforms for training and testing
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_federated_cifar100_dataloaders(
    num_clients,
    num_classes_per_client,
    batch_size=50,
    seed=42,
    federatedTest=False,
    val_split=0
):
    """Load CIFAR-100 dataset for federated learning scenario.
    Args:
        num_clients: Number of clients in the federated setup.
        num_classes_per_client: Number of classes assigned to each client.
        batch_size: Batch size for DataLoader.
        seed: Random seed for reproducibility.
        federatedTest: Whether to create separate test sets for each client.
        val_split: Fraction of training data to use for validation.
    Returns:
        train_datasets: List of DataLoader for each client's training data.
        val_loader: DataLoader for validation data.
        test_loaders: DataLoader for test data, either shared or
            separate for each client.
    """
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
    full_train_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_dataset = TransformedDataset(train_dataset, transform_train)
    val_dataset = TransformedDataset(val_dataset, transform_test)

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

    for num_class in range(1, num_classes_per_client + 1):
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

    total_samples = train_size
    samples_per_client = total_samples // num_clients

    for client_id, class_set in enumerate(client_class_map):
        per_class_quota = samples_per_client // num_classes_per_client
        for cls in class_set:
            available = [i for i in train_class_indices[cls] if i not in used_train_indices]
            if len(available) < per_class_quota:
                raise ValueError(f"Not enough samples in class {cls} for client {client_id}")
            selected = available[:per_class_quota]
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
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loaders.append(test_loader)
    else:
        test_loaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_datasets, val_loader, test_loaders, client_class_map

def create_mixed_bias_and_size_partition(
    num_clients=45,
    dataset=None,
    num_classes=100,
    class_coverage_modes=(("full", 10, 1.0), ("moderate", 10, 0.5), ("strong", 25, 0.2)),
    size_modes=("small", "medium", "large"),
    size_weights=(0.2, 0.6, 0.2),
    size_means=(500, 1000, 2000),
    size_stds=(50, 100, 200),
    beta=5.0,
    seed=1
):
    """Create a partition of the CIFAR-100 dataset with mixed class coverage and size.
    Args:
        num_clients: Number of clients to partition the dataset into.
        dataset: The CIFAR-100 dataset to partition.
        num_classes: Total number of classes in the dataset.
        class_coverage_modes: List of tuples (name, count, fraction) defining class coverage for clients.
        size_modes: List of size categories for clients.
        size_weights: Probability for each size category.
        size_means: Mean sizes for each size category.
        size_stds: Standard deviations for each size category.
        beta: Concentration parameter for Dirichlet distribution.
        seed: Random seed for reproducibility.
    Returns:
        client_indices: List of lists, where each sublist contains indices of samples for a client.
        client_class_map: List of sets, where each set contains classes assigned to a client.
        client_metadata: List of dictionaries containing metadata for each client.
    """
    rng = np.random.default_rng(seed)

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)
    total_class_samples = {c: rng.permutation(class_to_indices[c]).tolist() for c in range(100)}

    class_allocation_counter = defaultdict(int)
    client_indices = [[] for _ in range(num_clients)]
    client_class_map = [set() for _ in range(num_clients)]
    client_metadata = [{} for _ in range(num_clients)]
    client_sizes = np.zeros(num_clients, dtype=int)

    # Assign class coverage (bias)
    client_pool = list(range(num_clients))
    assigned = 0
    for bias_name, count, frac in class_coverage_modes:
        group = client_pool[assigned:assigned + count]
        assigned += count
        num_classes_per_client = int(num_classes * frac)
        for client_id in group:
            chosen_classes = rng.choice(num_classes, size=num_classes_per_client, replace=False)
            client_class_map[client_id] = set(chosen_classes)
            client_metadata[client_id]["bias"] = bias_name

    # Assign sizes
    size_labels = rng.choice(size_modes, size=num_clients, p=size_weights)
    for i, size_label in enumerate(size_labels):
        idx = size_modes.index(size_label)
        size = int(np.clip(rng.normal(size_means[idx], size_stds[idx]), 30, None))
        client_sizes[i] = size
        client_metadata[i]["size"] = size_label

    # Allocate samples per client based on their view
    for client_id in range(num_clients):
        size = client_sizes[client_id]
        classes = list(client_class_map[client_id])
        if not classes:
            continue

        proportions = rng.dirichlet([beta] * len(classes))
        class_alloc = (proportions * size).astype(int)

        for c, count in zip(classes, class_alloc):
            available = total_class_samples[c]
            used = class_allocation_counter[c]
            remaining = len(available) - used
            take = min(count, remaining)
            if take > 0:
                samples = available[used:used + take]
                client_indices[client_id].extend(samples)
                class_allocation_counter[c] += take

    # Fix remaining unallocated class samples
    for c, used_count in class_allocation_counter.items():
        available = total_class_samples[c]
        remaining = len(available) - used_count
        if remaining <= 0:
            continue

        leftover = available[used_count:]
        eligible_clients = [i for i in range(num_clients) if c in client_class_map[i]]
        if not eligible_clients:
            continue

        idx = 0
        for sample in leftover:
            client_id = eligible_clients[idx % len(eligible_clients)]
            client_indices[client_id].append(sample)
            idx += 1

    return client_indices, client_class_map, client_metadata


def get_unbalanced_cifar100_datasets(
    num_clients=45,
    class_coverage_modes=(("full", 10, 1.0), ("moderate", 10, 0.5), ("strong", 25, 0.2)),
    size_modes=("small", "medium", "large"),
    size_weights=(0.2, 0.6, 0.2),
    size_means=(500, 1000, 2000),
    size_stds=(50, 100, 200),
    beta=5.0,
    batch_size=50,
    seed=1
):
    """Load CIFAR-100 dataset with unbalanced partitioning for federated learning.
    Args:
        num_clients: Number of clients to partition the dataset into.
        class_coverage_modes: List of tuples (name, count, fraction) defining class coverage for clients.
        size_modes: List of size categories for clients.
        size_weights: Probability for each size category.
        size_means: Mean sizes for each size category.
        size_stds: Standard deviations for each size category.
        beta: Concentration parameter for Dirichlet distribution.
        batch_size: Batch size for DataLoader.
        seed: Random seed for reproducibility.
    Returns:
        client_datasets: List of datasets for each client.
        test_loader: DataLoader for test data.
        client_class_map: List of sets, where each set contains classes assigned to a client.
        client_metadata: List of dictionaries containing metadata for each client.
    """
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

    # Create mixed bias and size partition
    client_indices, client_class_map, client_metadata = create_mixed_bias_and_size_partition(
        num_clients=num_clients,
        dataset=train_dataset,
        class_coverage_modes=class_coverage_modes,
        size_modes=size_modes,
        size_weights=size_weights,
        size_means=size_means,
        size_stds=size_stds,
        beta=beta,
        seed=seed
    )

    # Create client datasets and test loader
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return client_datasets, test_loader, client_class_map, client_metadata