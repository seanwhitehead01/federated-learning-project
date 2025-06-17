import numpy as np
from collections import Counter

def compute_discrepancy(client_dataset, num_classes=100, max_classes=10, alpha=2.0):
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
