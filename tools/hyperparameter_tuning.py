import torch.optim as optim
from eval import evaluate
from train import train

def run_grid_search(train_loader, val_loader, model_fn, criterion, configs, device):
    best_acc = 0
    best_cfg = None
    best_model_state = None
    results = []

    for cfg in configs:
        model = model_fn(device)
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'])

        best_loss = float('inf')
        patience_counter = 0

        print(f"Training with lr={cfg['lr']}, momentum={cfg['momentum']}, T_max={cfg['T_max']}...")
        for epoch in range(10):  # Small number of epochs for quick grid search
            train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            if val_loss < best_loss - 1e-3:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print("  Early stopping")
                break

        results.append((cfg['lr'], cfg['momentum'], cfg['T_max'], val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = (cfg['lr'], cfg['momentum'], cfg['T_max'])
            best_model_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }


    print(f"Best val acc: {best_acc:.4f} with lr={best_cfg[0]}, momentum={best_cfg[1]}, T_max={best_cfg[2]}")
    return best_cfg, best_model_state, results