from collections import defaultdict
import torch
import torch.nn.functional as F
import copy
from math import floor

def fischer_scores(model, dataloader, device, R=1, mask=None, N=1, num_classes=100):
    model.eval()
    model.to(device)

    scores = {name: torch.zeros_like(p, device='cpu') for name, p in model.named_parameters() if p.requires_grad}
    class_counts = defaultdict(int)

    for data_batch, labels_batch in dataloader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)

        for i in range(data_batch.size(0)):
            x = data_batch[i].unsqueeze(0)
            y = labels_batch[i].item()

            if class_counts[y] >= N:
                continue
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
                        g = param.grad.detach().cpu()
                        if mask is not None and name in mask:
                            g = g * mask[name].float().cpu()
                        scores[name] += g ** 2

        if all(class_counts[c] >= N for c in range(num_classes)):
            break

    return scores

def mask_calculator(model, dataset, device, rounds=4, sparsity=0.1, R=1, samples_per_class=1, num_classes=100, verbose=True):
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    param_map = {name: p for name, p in model_copy.named_parameters() if p.requires_grad}
    mask = {name: torch.ones_like(p, dtype=torch.bool) for name, p in param_map.items()}

    for r in range(1, rounds + 1):
        if verbose:
            print(f"\n[Round {r}/{rounds}]")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=50,
            shuffle=True,
            num_workers=4
        )

        scores = fischer_scores(
            model_copy, dataloader, device=device, R=R,
            mask=mask, N=samples_per_class, num_classes=num_classes
        )

        # Set +inf to already masked or zero-score elements
        for name in scores:
            float_mask = mask[name].float()
            scores[name][float_mask == 0] = float('inf')
            scores[name][scores[name] == 0] = float('inf')

        # Flatten and rank scores
        all_scores = torch.cat([f.flatten() for f in scores.values()])
        m = all_scores.numel()
        keep_ratio = sparsity ** (r / rounds)
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

    # Freeze fully masked parameters
    threshold = 0.01
    masked_layers = 0
    original_param_map = {name: p for name, p in model.named_parameters() if p.requires_grad}
    for name, p in original_param_map.items():
        if mask[name].sum().item() / mask[name].numel() < threshold:
            p.requires_grad = False
            masked_layers += 1
            del mask[name]

    if verbose:
        print(f"  → {masked_layers} layers frozen due to masking (threshold = {threshold})")
    return mask
