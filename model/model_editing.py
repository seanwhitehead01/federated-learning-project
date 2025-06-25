from collections import defaultdict
import torch
import torch.nn.functional as F
import copy
from math import floor


def fischer_scores(model, dataloader, device, R=1, mask=None, N=None):
    """ Computes the Fisher scores for the model parameters based on the provided data loader.
    Args:
        model: The model for which to compute the Fisher scores.
        dataloader: DataLoader providing the data for computing scores.
        device: Device to run the computation on (CPU or GPU).
        R: Number of samples per data point to average over.
        mask: Optional mask to apply to gradients.
        N: Dictionary specifying the number of samples per class.
    Returns:
        Dictionary of Fisher scores for each parameter in the model.
    """
    model.eval()
    model.to(device)

    class_labels = [i for i, count in enumerate(N) if count > 0]

    # Initialize scores
    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    
    # Explicitly initialize class_counts
    class_counts = {c: 0 for c in class_labels}

    # Iterate over the data loader
    for data_batch, labels_batch in dataloader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)

        # Iterate over each sample in the batch
        for i in range(data_batch.size(0)):
            x = data_batch[i].unsqueeze(0)
            y = labels_batch[i].item()

            if y not in class_labels:
                continue  # skip irrelevant classes

            # Check if we have enough samples for this class
            if class_counts[y] >= N[y]:
                continue
            class_counts[y] += 1

            # Forward pass
            for _ in range(R):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                sampled_y = torch.multinomial(probs, 1).squeeze(1)
                log_probs = F.log_softmax(logits, dim=1)
                logp = log_probs[0, sampled_y]

                loss = -logp
                model.zero_grad()
                loss.backward(retain_graph=True)

                # Accumulate gradients  
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.detach()
                        if mask is not None and name in mask:
                            g = g * mask[name].float()
                        scores[name] += g ** 2

        # Check if we have enough samples for all classes
        if all(class_counts[c] >= N[c] for c in class_labels):
            break

    return scores

def mask_calculator(model, dataset, device, rounds=4, density=0.1, R=1, samples_per_class=None, verbose=True, batch_size=50):
    """ Calculates a mask for the model parameters based on Fisher scores.
    Args:
        model: The model for which to calculate the mask.
        dataset: Dataset to use for calculating the Fisher scores.
        device: Device to run the computation on (CPU or GPU).
        rounds: Number of rounds to perform the mask calculation.
        density: Desired density of the mask after all rounds.
        R: Number of samples per data point to average over.
        samples_per_class: Dictionary
            specifying the number of samples to use per class for Fisher score calculation.
        verbose: Whether to print progress information.
        batch_size: Batch size for the DataLoader.
    Returns:
        Dictionary of masks for each layer in the model.
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # Create a mask for each layer in the model
    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.bool) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        if verbose:
            print(f"\n[Round {r}/{rounds}]")

        # Create a DataLoader for the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Compute Fisher scores
        scores = fischer_scores(
            model_copy, dataloader, device=device, R=R,
            mask=mask, N=samples_per_class
        )

        # Set +inf to already masked or zero-score elements
        for name in scores:
            float_mask = mask[name].float()
            scores[name][float_mask == 0] = float('inf')
            scores[name][scores[name] == 0] = float('inf')

        # Flatten and rank scores
        all_scores = torch.cat([f.flatten() for f in scores.values()])
        m = all_scores.numel()
        keep_ratio = density ** (r / rounds)
        k = floor(m * keep_ratio)
        threshold = torch.kthvalue(all_scores, k).values.item()
        if verbose:
            print(f"  → Keeping top {k} parameters (threshold = {threshold:.2e})")

        # Update mask using bool logic
        for name in scores:
            new_mask = (scores[name] <= threshold)
            mask[name] = mask[name] & new_mask  # keep only previously active & newly selected

        # Apply hard pruning
        with torch.no_grad():
            for name, p in param_map.items():
                p.data *= mask[name].to(dtype=p.dtype, device=p.device)

    return mask

def freeze_and_clean_client_masks(model, client_mask_dict, threshold=0.01, K=100, verbose=True):
    """ Freezes parameters in the model that are not used by any client based on their masks.
    Args:
        model: The model whose parameters will be frozen.
        client_mask_dict: Dictionary mapping client IDs to their parameter masks.
        threshold: Threshold below which layers are considered frozen.
        K: Total number of clients.
        verbose: Whether to print information about frozen parameters.
    Returns:
        client_mask_dict: Updated dictionary with frozen parameters removed from all client masks.
        frozen_state: List of layer names that were frozen.
    """
    # Initialize shared mask with all False (fully masked)
    first_mask = client_mask_dict[0]
    shared_mask = {name: mask.clone() for name, mask in first_mask.items()}

    # Combine masks from all clients using logical OR
    for cid in range(K):
        client_mask = client_mask_dict[cid]
        for name in shared_mask:
            shared_mask[name] |= client_mask.get(name, torch.zeros_like(shared_mask[name], dtype=torch.bool))

    # Freeze layers not used by any client (i.e., shared_mask == 0)
    frozen_count = 0
    total_count = 0
    frozen_state = []

    for name, param in model.named_parameters():
        if name in shared_mask:
            keep_ratio = shared_mask[name].sum().item() / shared_mask[name].numel()
            if keep_ratio <= threshold:  # frozen if 0 or below threshold
                param.requires_grad_(False)
                frozen_count += 1
                frozen_state.append(name)
                for cid in range(K):
                    if name in client_mask_dict[cid]:
                        del client_mask_dict[cid][name]
            total_count += 1

    if verbose:
        print(f"→ Frozen {frozen_count}/{total_count} parameters (based on logical OR across clients)")
        print(f"→ Removed frozen params from all client masks to save memory")

    return client_mask_dict, frozen_state
