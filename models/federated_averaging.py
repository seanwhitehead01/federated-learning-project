import torch.optim as optim
from train import train_steps
import copy

def get_trainable_keys(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def train_on_client(client_id, model, train_loader, steps, criterion, device, mask=None):
    print(f"  Training on client {client_id + 1}")
    model_copy = copy.deepcopy(model)
    optimizer = optim.SGD(model_copy.parameters(), lr=0.01)
    train_loss, train_acc = train_steps(model_copy, train_loader, optimizer, criterion, device, steps, mask)
    print(f"  Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    return model_copy.state_dict(), train_loss, train_acc


def average_metrics(metrics, weights):
    return sum(w * m for w, m in zip(weights, metrics))


def average_models(models, weights, keys):
    averaged_state = {}
    for key in keys:
        averaged_state[key] = sum(weights[i] * models[i][key] for i in range(len(models)))
    return averaged_state
