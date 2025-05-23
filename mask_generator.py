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

    # model_copy = copy.deepcopy(model).to(device)
    # s = compute_fisher_diagonal(model_copy, dataloader, device, num_samples=100)
    # all_scores = torch.cat([v.flatten() for v in s.values()])
    # k = floor(len(all_scores) * (1 - sparsity))
    # threshold = torch.kthvalue(all_scores, k).values.item()

    
    # mask = {}
    # for name, score_tensor in s.items():
    #     mask[name] = (score_tensor <= threshold).float()

    # model_copy = copy.deepcopy(model).to(device)
    # model_copy.eval()
    
    # # Get trainable parameters
    # param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    # mask = {name: torch.ones_like(p, dtype=torch.float32) for name, p in param_map.items()}

    # for r in range(1, rounds + 1):
    #     print(f"\n[Round {r}/{rounds}]")

    #     # Step 1: Compute Fisher Scores (diagonal approximation)
    #     fisher = {name: torch.zeros_like(p) for name, p in param_map.items()}

    #     for inputs, _ in tqdm(dataloader, desc=f"Computing Fisher (Round {r})"):
    #         inputs = inputs.to(device)
    #         logits = model_copy(inputs)
    #         probs = F.softmax(logits, dim=1)
    #         sampled_y = torch.multinomial(probs, 1).squeeze(1)
    #         log_probs = F.log_softmax(logits, dim=1)
    #         log_probs_samples = log_probs[torch.arange(inputs.size(0)), sampled_y]

    #         loss = -log_probs_samples.mean()
    #         model_copy.zero_grad()
    #         loss.backward()

    #         for name, p in param_map.items():
    #             if p.grad is not None:
    #                 fisher[name] += (p.grad.detach() ** 2)

    #     # Step 2: Apply new sparsity level
    #     all_scores = torch.cat([f.flatten() for f in fisher.values()])
    #     m = all_scores.numel()
    #     keep_ratio = sparsity ** (r / rounds)
    #     k = floor(m * keep_ratio)

    #     threshold = torch.kthvalue(all_scores, k).values.item()
    #     print(f"  → Keeping top {k} params (threshold = {threshold:.2e})")

    #     # Step 3: Update masks
    #     for name in fisher:
    #         mask[name] = (fisher[name] <= threshold).float()

    #     # Step 4: Apply mask to model weights (hard mask)
    #     with torch.no_grad():
    #         for name, p in param_map.items():
    #             p.data *= mask[name]

    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    # Get trainable parameters
    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.float32) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}/{rounds}]")

        # Step 1: Compute Fisher Scores (diagonal approximation)
        fisher = fisher_elementwise_per_sample(model_copy, dataloader, device=device)
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

def fisher_elementwise_per_sample(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    # Initialize score accumulator
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

    for data, _ in tqdm(dataloader):

        data = data.to(device)
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        sampled_y = torch.multinomial(probs, 1).squeeze(1)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs_samples = log_probs[torch.arange(data.size(0)), sampled_y]

        # x = x.unsqueeze(0).to(device)  # single input as a batch
        # logits = model(x)

        # # Get pseudo-label from model's own prediction
        # probs = F.softmax(logits, dim=1)
        # sampled_y = torch.multinomial(probs, 1).squeeze()
        # log_probs = F.log_softmax(logits, dim=1)
        # logp = log_probs[0, sampled_y]

        for idx in range(data.size(0)):
            # Compute log-prob of sampled labels
            logp = log_probs_samples[idx]
            loss = -logp.mean()

            # Backward to get gradient
            model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (per-element)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2)

    return fisher