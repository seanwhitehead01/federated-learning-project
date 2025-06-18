import numpy as np
from collections import Counter
import torch

def compute_discrepancy(client_dataset, num_classes=100, max_classes=25, alpha=1.0):
    """
    Discrepancy between client distribution and ideal local uniform over seen classes.

    Returns 0 if client sees max_classes and is uniform over them.

    Args:
        client_dataset: PyTorch Dataset (e.g., Subset)
        num_classes: total number of global classes
        max_classes: desired number of classes a client should have
        alpha: penalty factor for seeing fewer classes

    Returns:
        discrepancy: float
    """
    label_counts = Counter()
    for _, label in client_dataset:
        label_counts[label] += 1

    total = sum(label_counts.values())
    if total == 0:
        return 0.0

    seen_classes = sorted(label_counts.keys())
    num_seen = len(seen_classes)

    # Client's actual distribution
    dist = np.zeros(num_classes)
    for label, count in label_counts.items():
        dist[label] = count / total

    # Ideal distribution: uniform over seen classes only
    target = np.zeros(num_classes)
    for label in seen_classes:
        target[label] = 1.0 / num_seen

    # L2 discrepancy
    base_l2 = np.linalg.norm(dist - target, ord=2)

    # Penalize if client sees fewer than max_classes
    coverage_penalty = (max_classes / num_seen) ** alpha if num_seen < max_classes else 1.0

    return base_l2 * coverage_penalty

def compute_client_weight(client_size, discrepancy, total_samples, a=0.4, b=0.1, type="relu", tau=0.01):
    """
    Computes the weight for a single client.

    Args:
        client_size (int): Number of samples for the client
        discrepancy (float): L2 discrepancy of the client
        total_samples (int): Total number of samples across all clients
        a, b: weighting coefficients
        type (str): 'relu' or 'sigmoid'
        tau (float): temperature for sigmoid (only used if type='sigmoid')

    Returns:
        weight (float)
    """
    client_proportion = client_size / total_samples
    score = client_proportion - a * discrepancy + b

    if type == "relu":
        weight = torch.relu(torch.tensor(score)).item()
    elif type == "sigmoid":
        score /= tau
        weight = torch.sigmoid(torch.tensor(score)).item()
    else:
        raise ValueError("type must be 'relu' or 'sigmoid'")

    return weight

def get_class_distribution_vector(client_dataset, num_classes=100):
    """
    Returns the per-class proportion of samples in the client's dataset.

    Args:
        client_dataset: PyTorch Dataset or Subset
        num_classes: total number of classes (default = 100 for CIFAR-100)

    Returns:
        distribution: list of floats of length num_classes, summing to 1
    """
    label_counts = Counter()

    for _, label in client_dataset:
        label_counts[label] += 1

    total = sum(label_counts.values())
    if total == 0:
        return [0.0] * num_classes

    distribution = [label_counts[c] / total for c in range(num_classes)]
    return distribution