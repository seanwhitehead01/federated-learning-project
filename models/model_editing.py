import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import floor
from tqdm import tqdm

# This function computes the un-normalized Fisher scores for each parameter in the model
def fischer_scores(model, dataloader, device, R=1, mask=None):
    model.eval()
    model.to(device)

    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

    for data, _ in tqdm(dataloader, desc="Scoring"):
        data = data.to(device)

        for _ in range(R):
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            sampled_y = torch.multinomial(probs, 1).squeeze(1)
            log_probs = F.log_softmax(logits, dim=1)
            logp = log_probs[torch.arange(data.size(0)), sampled_y]

            for i in range(data.size(0)):
                loss = -logp[i]
                model.zero_grad()
                loss.backward(retain_graph=True)

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.detach()
                        if mask is not None and name in mask:
                            g = g * mask[name].to(g.device)
                        scores[name] += g ** 2

    return scores

def mask_calculator(model, dataloader, device, rounds=5, sparsity=0.5, R=1):
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # Initialize masks and param references
    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.float32) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}/{rounds}]")

        # Step 1: Compute Fisher scores using current mask
        scores = fischer_scores(model_copy, dataloader, device=device, R=R, mask=mask)

        # Step 2: Set scores of already masked params to +inf
        for name in scores:
            scores[name][mask[name] == 0] = float('inf')

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