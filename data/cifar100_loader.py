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
    num_classes_per_client=10,
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

def get_federated_cifar100_dataloaders_with_imbalances(
    num_clients=100,
    num_classes_per_client=10,
    batch_size=50,
    seed=0,
    size_unbalanced=True,
    class_unbalanced=True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

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

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        class_to_indices[label].append(idx)
    for cls in class_to_indices:
        rng.shuffle(class_to_indices[cls])


    # Configurations
    size_types = {
        "large": {"mean": 1000, "std": 100},
        "medium": {"mean": 500, "std": 50},
        "small": {"mean": 100, "std": 20},
    }
    size_probs = [0.2, 0.6, 0.2]
    distribution_types = ["uniform", "small_unbalance", "large_unbalance"]
    dist_probs = [0.6, 0.25, 0.15]


    # Assign size and class distribution per client
    assigned_sizes = []
    assigned_distributions = []
    for _ in range(num_clients):
        if size_unbalanced:
            size_type = rng.choice(list(size_types.keys()), p=size_probs)
        else:
            size_type = "medium"
        assigned_sizes.append(size_type)

        if class_unbalanced:
            dist_type = rng.choice(distribution_types, p=dist_probs)
        else:
            dist_type = "uniform"
        assigned_distributions.append(dist_type)

    # Convert size labels to sample counts
    client_sizes = []
    for size_type in assigned_sizes:
        cfg = size_types[size_type]
        s = int(np.clip(rng.normal(cfg["mean"], cfg["std"]), 30, None)) # No less than 30 samples
        client_sizes.append(s)

    class_probs = np.ones(100)
    class_probs /= class_probs.sum()

    # Decay
    alpha = 0.9  # Decay factor for class probabilities
    # Gamma parameters (unchanged)
    gamma_base_penalty = 0.7
    gamma_penalty_size = 0.7
    gamma_reward_size = 1.3
    gamma_penalty_small = 0.9
    gamma_reward_small = 1.3
    gamma_penalty_large = 0.1
    gamma_reward_large = 1.7


    client_class_map = []
    client_indices = [[] for _ in range(num_clients)]


    sample_per_class = defaultdict(int)  # ← moved up to track class samples in real-time

    for i in range(num_clients):
        size_type = assigned_sizes[i]
        dist_type = assigned_distributions[i]

        # Compute imbalance penalty
        current_counts = np.array([sample_per_class.get(cls, 0) for cls in range(100)])
        imbalance_penalty = np.exp(current_counts / (current_counts.max() +  1e-9))  # Avoid division by zero
        imbalance_penalty /= imbalance_penalty.sum()

        # Adjust class_probs using penalty
        adjusted_probs = class_probs / imbalance_penalty
        adjusted_probs /= adjusted_probs.sum()

        # Sample classes with adjusted probs
        chosen_classes = rng.choice(100, num_classes_per_client, replace=False, p=adjusted_probs)
        client_class_map.append(set(chosen_classes))

        updated_probs = class_probs.copy()

        # Update class_probs according to client type and selected classes
        if size_type == 'large':
            base_variation = gamma_base_penalty * gamma_penalty_size 
        elif size_type == 'small':
            base_variation = gamma_base_penalty * gamma_reward_size 
        else:
            base_variation = gamma_base_penalty 

        if dist_type == 'large_unbalance':
            updated_probs[chosen_classes[0]] *= gamma_penalty_large
            updated_probs[chosen_classes[1:]] *= gamma_reward_large
        elif dist_type == 'small_unbalance':
            half = len(chosen_classes) // 2
            updated_probs[chosen_classes[:half]] *= gamma_penalty_small
            updated_probs[chosen_classes[half:]] *= gamma_reward_small

        updated_probs[chosen_classes] *= base_variation
        updated_probs /= updated_probs.sum()

        class_probs = alpha * updated_probs + (1 - alpha) * class_probs
        class_probs /= class_probs.sum()
        
        # Simulate real sample allocation for global tracking
        class_ids = list(chosen_classes)
        size = client_sizes[i]
        alloc = allocate_samples(class_ids, size, dist_type)

        for cls, count in alloc.items():
            available = class_to_indices[cls]
            available = rng.permutation(available).tolist()  # Shuffle to avoid always picking the same ones

            if len(available) >= count:
                # Use only unique samples
                selected = available[:count]
            else:
                # Use all unique samples first
                selected = available.copy()
                # Fill the rest with random duplicates
                extra_needed = count - len(available)
                extra_samples = rng.choice(available, size=extra_needed, replace=True).tolist()
                selected.extend(extra_samples)

            client_indices[i].extend(selected)

    train_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_datasets, test_loader, client_class_map, assigned_sizes, assigned_distributions

import numpy as np
from collections import Counter

def get_class_distribution_matrix(train_datasets, client_metadata=None, verbose=False):
    """
    Returns a matrix of shape (num_clients, 100) where entry (i, j) is the number of samples
    of class j held by client i. Optionally prints the distribution per client.

    Args:
        train_datasets: list of torch.utils.data.Dataset per client
        client_metadata: list of dicts with 'size_type' and 'class_distribution_type' per client (optional)
        verbose: whether to print summary table
    
    Returns:
        dist_matrix: np.ndarray of shape (num_clients, 100)
    """
    num_clients = len(train_datasets)
    num_classes = 100
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)

    for client_id, dataset in enumerate(train_datasets):
        label_counts = Counter()
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_counts[label] += 1

        for label, count in label_counts.items():
            dist_matrix[client_id, label] = count

        if verbose:
            meta_str = ""
            if client_metadata:
                meta = client_metadata[client_id]
                meta_str = f"| Size: {meta['size_type']:<6} | Dist: {meta['class_distribution_type']:<15} "
            summary = ", ".join(f"{cls}:{cnt}" for cls, cnt in sorted(label_counts.items()))
            print(f"Client {client_id:>3} {meta_str}| {summary}")

    return dist_matrix
        
import matplotlib.pyplot as plt

def plot_client_distributions(train_datasets, client_metadata, num_clients_to_plot=6):
    """
    Plots class distributions for a few selected clients.
    
    Args:
        train_datasets: List of datasets for each client.
        client_metadata: List of dicts with metadata per client.
        num_clients_to_plot: How many clients to visualize.
    """
    selected_clients = list(range(min(num_clients_to_plot, len(train_datasets))))
    fig, axs = plt.subplots(len(selected_clients), 1, figsize=(10, 3 * len(selected_clients)))
    
    if len(selected_clients) == 1:
        axs = [axs]
    
    for ax, client_id in zip(axs, selected_clients):
        dataset = train_datasets[client_id]
        label_counts = Counter()
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_counts[label] += 1

        classes = sorted(label_counts.keys())
        counts = [label_counts[c] for c in classes]

        ax.bar(classes, counts)
        ax.set_title(f"Client {client_id} — Size: {client_metadata[client_id]['size_type']}, "
                     f"Dist: {client_metadata[client_id]['class_distribution_type']}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Sample Count")
        ax.set_xticks(classes)
    
    plt.tight_layout()
    plt.show()

