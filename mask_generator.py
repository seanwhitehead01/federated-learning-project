import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import grad
from models.federated_averaging import get_trainable_keys
from math import floor
import copy


def score(model, dataloader, mask, device): 
    softmax = nn.Softmax(dim=1)
    trainable_keys = get_trainable_keys(model)
    s = {key: 0 for key in trainable_keys} 
    
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        logits = model(inputs)  
        probs = softmax(logits)  
        samples = torch.multinomial(probs, num_samples=1).squeeze(1)
        log_probs = F.log_softmax(logits, dim=1)
        batch_idx = torch.arange(inputs.size(0), device=device)
        log_prob_sampled = log_probs[batch_idx, samples]     
        model.zero_grad()
        log_prob_sampled.backward() 
            
        for name, param in model.named_parameters(): # key == name
            if name in s:  
                s[name] = s[name] + param.grad ** 2
                
    for name in mask:
        if mask[name] == 0:
            s[name] = float('inf')
            
    return s
    
    
def mask_calculator(model, dataloader, rounds, sparsity, device):
    trainable_keys = get_trainable_keys(model) 
    model_copy = copy.deepcopy(model).to(device)
    m = len(trainable_keys)
    mask = {key: 1 for key in trainable_keys}  


    for r in range(rounds):        
        to_keep = sparsity ** (r / rounds)
        s = score(model_copy, dataloader, mask, device)
        
        p = floor(m*to_keep) 
        s_hat = sorted(s.items(), key=lambda item: item[1],  reverse=True)
        _, pth_value = s_hat[p - 1]
            
        for name in trainable_keys:
            if s[name] - pth_value > 0: 
                mask[name] = 0
            
        # parameter update - Hadamard Product 
        for name, param in model_copy.named_parameters():
            if name in mask: 
                param.data *= mask[name]  
    
    return mask 
