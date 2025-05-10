import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

def dataset_division(train_dataset, batch_size=256): 
    class_to_indices = defaultdict(list) # dizionario di liste 

    for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
        class_to_indices[label].append(idx)    
    
     # random shuffle of indices 
    for indices in class_to_indices.values(): # restituisce liste di indici per ogni classe 
        random.shuffle(indices)

    sharding_datasets = []
    cardinality_datasets = [] 
    total_indices = [[] for _ in range(100)]
        
    for class_id,indices in class_to_indices.items(): # class_id identifica dizionario, indices è lista 
        n = len(indices)
        chunk_size = n // 100
        remainder = n % 100 
        start = 0
        for i in range(100): 
            extra = 1 if i < remainder else 0
            end = start + chunk_size + extra
            total_indices[i].extend(indices[start:end]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
            start = end
    
    for i in range(100): 
        subset = torch.utils.data.Subset(train_dataset, total_indices[i])
        sharding_datasets.append(subset)

    sharding_dataloaders = []
    for dataset in sharding_datasets:
        cardinality_datasets.append(len(dataset)) # record of cardinality of datasets 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        sharding_dataloaders.append(loader)

    return sharding_dataloaders, cardinality_datasets 