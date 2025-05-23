import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import floor
from tqdm import tqdm

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# # TODO: fix score function (now it returns only zeros)
# def score(model, dataloader, mask, device, trainable_params):
#     softmax = nn.Softmax(dim=1)
#     s = {id(p): torch.zeros_like(p) for p in trainable_params}
    
#     model.eval()
#     for inputs, _ in tqdm(dataloader, desc="Scoring"):
#         inputs = inputs.to(device)

#         logits = model(inputs)
#         probs = softmax(logits)
#         samples = torch.multinomial(probs, num_samples=1).squeeze(1)

#         log_probs = F.log_softmax(logits, dim=1)
#         log_prob_sampled = log_probs[torch.arange(inputs.size(0), device=device), samples]
#         loss = log_prob_sampled.mean()  # average over batch
#         model.zero_grad()
#         loss.backward()

#         for p in trainable_params:
#             if p.grad is not None:
#                 s[id(p)] += (p.grad.detach() ** 2)

#     for p in trainable_params:
#         if id(p) in mask:
#             s[id(p)][mask[id(p)] == 0] = float('inf')  # Ignore pruned params

#     return s


def compute_fisher_diagonal(model, dataloader, device='cuda', num_samples=None):
    model.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    count = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)

        # Sample pseudo-labels from model prediction
        sampled_y = torch.multinomial(probs, 1).squeeze()

        # Compute log-prob of sampled labels
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_sampled = log_probs[torch.arange(len(sampled_y)), sampled_y]
        loss = -log_prob_sampled.mean()

        # Backward to get gradient
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += (param.grad.detach() ** 2)

        count += inputs.size(0)
        if num_samples and count >= num_samples:
            break

    # Normalize
    for name in fisher:
        fisher[name] /= count

    return fisher

def mask_calculator(model, dataloader, device, rounds=5, sparsity=0.5):
    # trainable_params = get_trainable_params(model) 
    # mask = {id(p): torch.ones_like(p) for p in trainable_params}

    # model_copy = copy.deepcopy(model).to(device)

    # for r in range(rounds):
    #     to_keep = sparsity ** ((r + 1) / rounds)
    #     s = score(model_copy, dataloader, mask, device, trainable_params)

    #     # Flatten all scores
    #     all_scores = torch.cat([v.flatten() for v in s.values()])
    #     k = floor(len(all_scores) * to_keep)
    #     print(f"Round {r + 1}/{rounds}, keeping {to_keep:.2%} of parameters, k={k}")
    #     threshold = torch.kthvalue(all_scores, k).values.item()
    #     print(f"Threshold: {threshold}")

    #     # Update masks
    #     for p in trainable_params:
    #         score_tensor = s[id(p)]
    #         new_mask = (score_tensor <= threshold).float()
    #         mask[id(p)] = new_mask

    #     # Apply new mask to model_copy
    #     with torch.no_grad():
    #         for p in model_copy.parameters():
    #             if id(p) in mask:
    #                 p.data *= mask[id(p)]

    model_copy = copy.deepcopy(model).to(device)
    s = compute_fisher_diagonal(model_copy, dataloader, device, num_samples=100)
    all_scores = torch.cat([v.flatten() for v in s.values()])
    k = floor(len(all_scores) * (1 - sparsity))
    threshold = torch.kthvalue(all_scores, k).values.item()

    
    mask = {}
    for name, score_tensor in s.items():
        mask[name] = (score_tensor <= threshold).float()

    return mask, s
