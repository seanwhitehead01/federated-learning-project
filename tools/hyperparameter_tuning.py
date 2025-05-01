import torch.optim as optim
import itertools
from eval import evaluate
from train import train

def run_grid_search(train_loader, val_loader, model_fn, criterion, device):
    lr_list = [0.01, 0.001]
    momentum_list = [0.9, 0.95]
    T_max_list = [10, 30]
    best_acc = 0
    best_cfg = None
    best_model_state = None

    for lr, momentum, T_max in itertools.product(lr_list, momentum_list, T_max_list):
        model = model_fn(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        print(f"Training with lr={lr}, momentum={momentum}, T_max={T_max}...")
        for epoch in range(5):  # Small number of epochs for quick grid search
            train(model, train_loader, optimizer, criterion, device)
            _, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

        print(f"Validation accuracy: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = (lr, momentum, T_max)
            best_model_state = model.state_dict()

    print(f"Best val acc: {best_acc:.4f} with lr={best_cfg[0]}, momentum={best_cfg[1]}, T_max={best_cfg[2]}")
    return best_cfg, best_model_state