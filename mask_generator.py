import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import floor
from tqdm import tqdm

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def score(model, dataloader, mask, device):
    softmax = nn.Softmax(dim=1)
    trainable_params = get_trainable_params(model)
    s = {id(p): torch.zeros_like(p) for p in trainable_params}
    
    model.eval()
    for inputs, _ in tqdm(dataloader, desc="Scoring"):
        inputs = inputs.to(device)

        logits = model(inputs)
        probs = softmax(logits)
        samples = torch.multinomial(probs, num_samples=1).squeeze(1)

        log_probs = F.log_softmax(logits, dim=1)
        log_prob_sampled = log_probs[torch.arange(inputs.size(0), device=device), samples]
        loss = log_prob_sampled.mean()  # average over batch
        model.zero_grad()
        loss.backward()

        for p in trainable_params:
            if p.grad is not None:
                s[id(p)] += (p.grad.detach() ** 2)

    for p in trainable_params:
        if id(p) in mask:
            s[id(p)][mask[id(p)] == 0] = float('inf')  # Ignore pruned params

    return s

def mask_calculator(model, dataloader, device, rounds=5, sparsity=0.5):
    trainable_params = get_trainable_params(model)
    mask = {id(p): torch.ones_like(p) for p in trainable_params}

    model_copy = copy.deepcopy(model).to(device)

    for r in range(rounds):
        to_keep = sparsity ** (r / rounds)
        s = score(model_copy, dataloader, mask, device)

        # Flatten all scores
        all_scores = torch.cat([v.flatten() for v in s.values()])
        k = floor(len(all_scores) * to_keep)
        threshold = torch.kthvalue(all_scores, k).values.item()

        # Update masks
        for p in trainable_params:
            score_tensor = s[id(p)]
            new_mask = (score_tensor <= threshold).float()
            mask[id(p)] = new_mask

        # Apply new mask to model_copy
        with torch.no_grad():
            for p in model_copy.parameters():
                if id(p) in mask:
                    p.data *= mask[id(p)]

    return mask
