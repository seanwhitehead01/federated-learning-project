# This script defines a function to load the CIFAR-100 dataset using PyTorch.
# It includes data augmentation and normalization transformations for training and testing sets.
# The function returns DataLoader objects for both training and testing datasets.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from collections import defaultdict
import random
import torch
import os
import pickle
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


# Federated CIFAR-100 Dataloaders
# This function creates federated dataloaders for CIFAR-100 dataset.
def get_federated_cifar100_dataloaders(
    num_clients,
    num_classes_per_client,
    batch_size=50,
    seed=42,
    class_balanced=True,
    federatedTest=False,
    val_split=0
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
            
            if len(all_available) >= samples_per_client:
                random.shuffle(all_available)
                selected = all_available[:samples_per_client]
            else:
                # Not enough unique, fill with duplicates from allowed classes
                selected = all_available.copy()
                needed = samples_per_client - len(all_available)
                # Sample with replacement from assigned classes
                candidate_pool = []
                for cls in class_set:
                    candidate_pool += train_class_indices[cls]
                extra = random.choices(candidate_pool, k=needed)
                selected.extend(extra)

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

def allocate_samples(class_ids, total_samples, dist_type):
    alloc = defaultdict(int)
    n = len(class_ids)
    if dist_type == "uniform":
        per = total_samples // n
        for c in class_ids:
            alloc[c] = per
    elif dist_type == "small_unbalance":
        half = n // 2
        major = class_ids[:half]
        minor = class_ids[half:]
        major_total = int(total_samples * 0.8)
        minor_total = total_samples - major_total
        for c in major:
            alloc[c] = max(1, major_total // len(major))
        for c in minor:
            alloc[c] = max(1, minor_total // len(minor))
    elif dist_type == "large_unbalance":
        major = class_ids[0]
        minor = class_ids[1:]
        major_total = int(total_samples * 0.8)
        minor_total = total_samples - major_total
        alloc[major] = major_total
        for c in minor:
            alloc[c] = max(1, minor_total // len(minor))
    return alloc

def get_federated_cifar100_dataloaders_with_dirichlet(
    num_clients=100,
    num_classes_per_client=25,
    beta=1.0,
    batch_size=50,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Transforms
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

    # Load CIFAR-100
    train_dataset = datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)

    # Organize indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        class_to_indices[label].append(idx)

    # Shuffle indices within each class
    for cls in class_to_indices:    
        class_to_indices[cls] = np.random.permutation(class_to_indices[cls]).tolist()

    # Assign N classes to each client
    all_classes = np.arange(100)
    client_class_map = []
    for _ in range(num_clients):
        selected_classes = np.random.choice(all_classes, size=num_classes_per_client, replace=False)
        client_class_map.append(set(selected_classes))

    # Build reverse map: class → list of clients that have that class
    class_client_map = defaultdict(list)
    for client_id, class_set in enumerate(client_class_map):
        for cls in class_set:
            class_client_map[cls].append(client_id)

    # Assign samples using Dirichlet
    client_indices = [[] for _ in range(num_clients)]

    for cls, indices in class_to_indices.items():
        clients_with_class = class_client_map[cls]
        if not clients_with_class:
            continue  # Skip classes not assigned to any client

        proportions = np.random.dirichlet([beta] * len(clients_with_class))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(np.array(indices), proportions)

        for i, client_id in enumerate(clients_with_class):
            client_indices[client_id].extend(splits[i].tolist())

    # Wrap into torch.utils.data.Subset
    client_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return  client_datasets, test_loader, client_class_map

def create_mixed_bias_and_size_partition(
    num_clients=45,
    num_classes=100,
    samples_per_class=500,
    class_coverage_modes=(("strong", 25, 0.2), ("moderate", 10, 0.5), ("full", 10, 1.0)),
    size_modes=("small", "medium", "large"),
    size_weights=(0.2, 0.6, 0.2),
    size_means=(500, 1000, 2000),
    size_stds=(50, 100, 200),
    beta=5.0,
    seed=69
):
    rng = np.random.default_rng(seed)
    total_class_samples = {c: list(range(c * samples_per_class, (c + 1) * samples_per_class)) for c in range(num_classes)}
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


def create_niid2_cifar100_datasets(
    num_clients=45,
    class_coverage_modes=(("strong", 25, 0.2), ("moderate", 10, 0.5), ("full", 10, 1.0)),
    size_modes=("small", "medium", "large"),
    size_weights=(0.2, 0.6, 0.2),
    size_means=(500, 1000, 2000),
    size_stds=(50, 100, 200),
    beta=5.0,
    batch_size=50,
    seed=42
):
    # Transforms
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

    client_class_map, client_indices, client_metadata = create_mixed_bias_and_size_partition(
        num_clients=num_clients,
        class_coverage_modes=class_coverage_modes,
        size_modes=size_modes,
        size_weights=size_weights,
        size_means=size_means,
        size_stds=size_stds,
        beta=beta,
        seed=seed
    )

    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return client_datasets, test_loader, client_class_map, client_metadata