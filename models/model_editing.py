import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import floor
from tqdm import tqdm
from collections import defaultdict

# This function computes the un-normalized Fisher scores for each parameter in the model
def fischer_scores(model, dataloader, device, R=1, mask=None, N=1, num_classes=100):
    model.eval()
    model.to(device)

    # Initialize fisher scores
    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

    # Track how many samples per class have been processed
    class_counts = defaultdict(int)

    for data_batch, labels_batch in tqdm(dataloader, desc="Scoring"):
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)

        for i in range(data_batch.size(0)):
            x = data_batch[i].unsqueeze(0)  # Single sample
            y = labels_batch[i].item()

            if class_counts[y] >= N:
                continue  # Skip if already reached N for this class

            class_counts[y] += 1

            for _ in range(R):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                sampled_y = torch.multinomial(probs, 1).squeeze(1)
                log_probs = F.log_softmax(logits, dim=1)
                logp = log_probs[0, sampled_y]

                loss = -logp
                model.zero_grad()
                loss.backward(retain_graph=True)

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.detach()
                        if mask is not None and name in mask:
                            g = g * mask[name].to(g.device)
                        scores[name] += g ** 2

        # Check if we have collected enough samples
        if all(class_counts[c] >= N for c in range(num_classes)):
            break

    return scores

def mask_calculator(model, dataset, device, rounds=4, sparsity=0.1, R=1, samples_per_class=1, num_classes=100):
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # Initialize masks and param references
    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.float32) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}/{rounds}]")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=50,
            shuffle=True,
            num_workers=4
        )

        # Step 1: Compute Fisher scores using current mask
        scores = fischer_scores(model_copy, dataloader, device=device, R=R, mask=mask, 
                                N=samples_per_class, num_classes=num_classes)

        # Step 2: Set scores of already masked params to +inf
        for name in scores:
            # Masked out
            scores[name][mask[name] == 0] = float('inf')
            # Zero score = unused = not important
            scores[name][scores[name] == 0] = float('inf')

        # Step 3: Rank parameters and determine new threshold
        all_scores = torch.cat([f.flatten() for f in scores.values()])
        m = all_scores.numel()
        keep_ratio = sparsity ** (r / rounds)
        k = floor(m * keep_ratio)
        threshold = torch.kthvalue(all_scores, k).values.item()
        print(f"  → Keeping top {k} parameters (threshold = {threshold:.2e})")

        # Step 4: Update mask
        for name in scores:
            new_mask = (scores[name] <= threshold).float()
            mask[name] = mask[name] * new_mask  # Progressive shrinking

        # Step 5: Apply mask to weights (hard pruning)
        with torch.no_grad():
            for name, p in param_map.items():
                p.data *= mask[name]

    return mask