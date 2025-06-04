import torch.optim as optim
from eval import evaluate
from train import train
import random
from models.federated_averaging import get_trainable_keys, average_models, average_metrics, train_on_client

def run_grid_search(train_loader, val_loader, model_fn, criterion, configs, device):
    best_acc = 0
    best_cfg = None
    results = {'scheduler': [], 'lr': [], 'momentum': [], 'val_loss': [], 'val_acc': []}

    for cfg in configs:
        model = model_fn(device)
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        
        if cfg['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        elif cfg['scheduler'] == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=5)
        elif cfg['scheduler'] == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            raise ValueError("Unsupported scheduler type")

        best_loss = float('inf')
        patience_counter = 0

        print(f"Training with lr={cfg['lr']}, momentum={cfg['momentum']}, scheduler={cfg['scheduler']}")
        for epoch in range(5):  # Small number of epochs for quick grid search
            train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            if val_loss < best_loss - 1e-2:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 3:
                print("  Early stopping")
                break

        results['scheduler'].append(cfg['scheduler'])
        results['lr'].append(cfg['lr'])
        results['momentum'].append(cfg['momentum'])
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg


    print(f"Best val acc: {best_acc:.4f} with scheduler = {best_cfg['scheduler']}, lr={best_cfg['lr']}, momentum={best_cfg['momentum']}")
    best_results = {'cfg': best_cfg, 'accuracy': best_acc}
    return best_results, results

def run_grid_search_federated(train_datasets, val_loader, model_fn, criterion, configs, num_clients, C, steps, device):
    best_acc = 0
    best_cfg = None
    results = {'lr': [],'val_loss': [], 'val_acc': []}

    for cfg in configs:
        model = model_fn(device)

        print(f"Training with lr={cfg['lr']}")
        for round in range(10):  # Small number of epochs for quick grid search
            print(f"\n--- Round {round + 1}/10 ---")

            # Select clients
            selected_clients = random.sample(range(num_clients), int(C * num_clients))

            # Local training
            local_models, train_losses, train_accs = [], [], []
            for client_id in selected_clients:
                model_state, loss, acc = train_on_client(
                    client_id,
                    model,
                    train_datasets[client_id],
                    steps,
                    criterion,
                    cfg['lr'],
                    device
                )
                local_models.append(model_state)
                train_losses.append(loss)
                train_accs.append(acc)

            # Weighting by dataset size
            client_sample_counts = [len(train_datasets[c]) for c in selected_clients]
            total_samples = sum(client_sample_counts)
            client_weights = [count / total_samples for count in client_sample_counts]

            # Federated averaging
            trainable_keys = get_trainable_keys(model)
            averaged_state = average_models(local_models, client_weights, trainable_keys)
            new_state = model.state_dict()
            for key in averaged_state:
                new_state[key] = averaged_state[key]
            model.load_state_dict(new_state)

            # Log average training metrics
            avg_train_loss = average_metrics(train_losses, client_weights)
            avg_train_acc = average_metrics(train_accs, client_weights)
            print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_acc:.4f}")

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
            print(f"Avg Test Loss: {val_loss:.4f}, Avg Test Accuracy: {val_acc:.4f}")

        results['lr'].append(cfg['lr'])
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg

    print(f"Best val acc: {best_acc:.4f} with lr={best_cfg['lr']}")
    best_results = {'cfg': best_cfg, 'accuracy': best_acc}
    return best_results, results