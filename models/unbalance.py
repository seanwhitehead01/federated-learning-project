import numpy as np
from collections import Counter

def compute_l2_discrepancy(client_dataset, num_classes=100):
    """
    Computes the L2 discrepancy between a client's class distribution
    and the uniform distribution.

    Args:
        client_dataset: a PyTorch Dataset (e.g., Subset of CIFAR-100 for a client)
        num_classes: total number of classes (default: 100 for CIFAR-100)

    Returns:
        discrepancy (float): L2 distance to the uniform distribution
    """
    # Count class occurrences
    label_counts = Counter()
    for _, label in client_dataset:
        label_counts[label] += 1

    total_samples = len(client_dataset)
    if total_samples == 0:
        return 0.0  # avoid division by zero

    # Normalize to probability distribution
    client_dist = np.zeros(num_classes)
    for label, count in label_counts.items():
        client_dist[label] = count / total_samples

    # Uniform distribution over classes
    uniform_dist = np.ones(num_classes) / num_classes

    # L2 discrepancy
    discrepancy = np.linalg.norm(client_dist - uniform_dist, ord=2)
    return discrepancy
