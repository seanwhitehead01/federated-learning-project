import numpy as np
from collections import Counter
import torch

def get_class_distribution_vector(client_dataset, num_classes=100):
    label_counts = Counter()

    for _, label in client_dataset:
        label_counts[label] += 1

    total = sum(label_counts.values())
    if total == 0:
        return [0.0] * num_classes

    distribution = [label_counts[c] / total for c in range(num_classes)]
    return distribution

def compute_kl_discrepancy(client_dataset, num_classes=100, epsilon=1e-12):
    label_counts = Counter()
    for _, label in client_dataset:
        label_counts[label] += 1

    total = sum(label_counts.values())
    if total == 0:
        return 0.0  # avoid division by zero

    # Empirical distribution (P)
    p = np.zeros(num_classes)
    for label, count in label_counts.items():
        p[label] = count / total

    # Uniform distribution (Q)
    q = np.ones(num_classes) / num_classes

    # Add smoothing
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    kl_div = np.sum(p * np.log(p / q))
    return kl_div

def compute_discrepancy_aware_weights_sigmoid(client_sizes, discrepancies, a=0.4, b=0.01, tau=0.01):
    client_sizes = np.array(client_sizes, dtype=np.float64)
    discrepancies = np.array(discrepancies, dtype=np.float64)

    # Normalize size and discrepancy
    size_norm = client_sizes / client_sizes.sum()
    discrepancy_norm = discrepancies / (discrepancies.max() + 1e-9)

    # Compute logits for sigmoid
    logits = (size_norm - a * discrepancy_norm + b) / tau
    weights = 1 / (1 + np.exp(-logits))  # Sigmoid activation

    # Normalize to sum to 1
    weights /= weights.sum()
    return weights

def compute_severity(client_sizes, discrepancies, a=0.9, b=0.1):
    client_sizes = np.array(client_sizes, dtype=np.float64)
    discrepancies = np.array(discrepancies, dtype=np.float64)

    # Normalize size and discrepancy
    size_norm = client_sizes / (client_sizes.max() + 1e-9)
    discrepancy_norm = discrepancies / (discrepancies.max() + 1e-9)

    severity = a * discrepancy_norm + b * (1 - size_norm)

    return severity
