import random
from collections import defaultdict
import math

import torch
from torch.utils.data import DataLoader, Subset

def imbalance_division(train_dataset, type_sharding, batch_size=256): 
    # type_sharding = 1 : one class per client
    # type_sharding = 5 : five classes per client
    # type_sharding = 10 : ten classes per client 
    # type_sharding = 50 : fifty classes per client

    sharding_datasets = []
    cardinality_datasets = [] 
    sharding_dataloaders = []
    total_indices = [[] for _ in range(100)]
    counter_classes = [0 for _ in range(100)] # tiene conto del numero di classi per dataset 

    if type_sharding == 1: 
        class_to_indices = defaultdict(list) # dizionario di liste 

        for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
            class_to_indices[label].append(idx)
        
        for class_id, indices in class_to_indices.item(): 
            subset = torch.utils.data.Subset(train_dataset,indices) 
            sharding_datasets.append(subset)
    
    if type_sharding == 5 or type_sharding == 10 or type_sharding == 50: 
        class_to_indices = defaultdict(list) # dizionario di liste 

        for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
            class_to_indices[label].append(idx)

        for indices in class_to_indices.values(): # restituisce liste di indici per ogni classe 
            random.shuffle(indices)

        for class_id,indices in class_to_indices.items(): # class_id identifica dizionario, indices è lista 
            total = sum(counter_classes) 
            bound =  math.floor(total/100) 
            n = len(indices) # indices of samples associated to a specific class 
            chunk_size = n // type_sharding
            remainder = n % type_sharding
            start = 0
            visited_clients = [] # visited clients to avoid repetition - reset at each iteration  
            valid_indices = [i for i, x in enumerate(counter_classes) if x < (bound + 1/2)]
      
            for j in range(type_sharding): 
                random_index = random.choice(valid_indices)
                valid_indices.remove(random_index)
                visited_clients.append(random_index)
                counter_classes[random_index] += 1 

                extra = 1 if j < remainder else 0
                end = start + chunk_size + extra
                total_indices[random_index].extend(indices[start:end]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
                start = end
                
        for i in range(100): 
            subset = torch.utils.data.Subset(train_dataset, total_indices[i])
            sharding_datasets.append(subset)
              
    
        for dataset in sharding_datasets:
            cardinality_datasets.append(len(dataset)) # record of cardinality of datasets 
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            sharding_dataloaders.append(loader)

    return sharding_dataloaders, cardinality_datasets, counter_classes, sharding_datasets 