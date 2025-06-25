import torch
import torch.optim as optim
from train import train_steps
import copy

def get_trainable_keys(model):
    """ Returns the names of the parameters in the model that are trainable.
    Args:
        model: The PyTorch model to inspect.
    Returns:
        A set of parameter names that are trainable.
    """
    return {name for name, param in model.named_parameters() if param.requires_grad}


def train_on_client(client_id, model, train_dataset, steps, criterion, lr, device, mask=None, batch_size=50):
    """Trains the model on a single client.
    Args:
        client_id: Identifier for the client.
        model: The PyTorch model to train.
        train_dataset: The training dataset for the client.
        steps: Number of training steps to perform.
        criterion: Loss function to use.
        lr: Learning rate for the optimizer.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        mask: Optional mask for selective training.
        batch_size: Batch size for training.
    Returns:
        A tuple containing the model's state dictionary, training loss, and training accuracy.
    """
    print(f"  Training on client {client_id + 1}")
    
    # Create DataLoader for the client's training dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    model_copy = copy.deepcopy(model)
    optimizer = optim.SGD(model_copy.parameters(), lr=lr)

    # Train the model for a specified number of steps
    train_loss, train_acc = train_steps(model_copy, train_loader, optimizer, criterion, device, steps, mask)
    print(f"  Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    return model_copy.state_dict(), train_loss, train_acc


def average_metrics(metrics, weights):
    """Averages the metrics using the provided weights.
    Args:
        metrics: List of metrics to average.
        weights: List of weights corresponding to each metric.
    Returns:
        The weighted average of the metrics.
    """
    return sum(w * m for w, m in zip(weights, metrics))


def average_models(models, weights, keys):
    """Averages the models' state dictionaries using the provided weights.
    Args:
        models: List of model state dictionaries to average.
        weights: List of weights corresponding to each model.
        keys: List of keys in the state dictionaries to average.
    Returns:
        A new state dictionary that is the weighted average of the input models.
    """
    averaged_state = {}
    for key in keys:
        averaged_state[key] = sum(weights[i] * models[i][key] for i in range(len(models)))
    return averaged_state
