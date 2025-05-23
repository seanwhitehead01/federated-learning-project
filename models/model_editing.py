import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import floor
from tqdm import tqdm

# This function computes the un-normalized Fisher scores for each parameter in the model
def fischer_scores(model, dataloader, device, R=1):
    model.eval()

    # Initialize score accumulator
    scores = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

    for data, _ in tqdm(dataloader):

        data = data.to(device)

        for _ in range(R):
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            sampled_y = torch.multinomial(probs, 1).squeeze(1)
            log_probs = F.log_softmax(logits, dim=1)
            log_probs_samples = log_probs[torch.arange(data.size(0)), sampled_y]

            for idx in range(data.size(0)):
                # Compute log-prob of sampled labels
                loss = -log_probs_samples[idx]

                # Backward to get gradient
                model.zero_grad()
                loss.backward(retain_graph=True)

                # Accumulate squared gradients (per-element)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        scores[name] += (param.grad.detach() ** 2)

    return scores

def mask_calculator(model, dataloader, device, rounds=5, sparsity=0.5, R=1):

    model_copy = copy.deepcopy(model).to(device)

    # Get trainable parameters
    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.float32) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}/{rounds}]")

        # Step 1: Compute Fisher Scores (diagonal approximation)
        fisher = fischer_scores(model_copy, dataloader, device, R=R)
        # Step 2: Apply new sparsity level
        all_scores = torch.cat([f.flatten() for f in fisher.values()])
        m = all_scores.numel()
        keep_ratio = sparsity ** (r / rounds)
        k = floor(m * keep_ratio)

        threshold = torch.kthvalue(all_scores, k).values.item()
        print(f"  → Keeping top {k} params (threshold = {threshold:.2e})")

        # Step 3: Update masks
        for name in fisher:
            mask[name] = (fisher[name] <= threshold).float()

        # Step 4: Apply mask to model weights (hard mask)
        with torch.no_grad():
            for name, p in param_map.items():
                p.data *= mask[name]

    return mask