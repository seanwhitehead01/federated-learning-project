import torch

def train(model, loader, optimizer, criterion, device, grad_mask=None):
    """ Trains the model for one epoch.
    Args:
        model: The model to train.
        loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the training on (CPU or GPU).
        grad_mask: Optional mask to apply to gradients.
    Returns:
        Tuple of average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct = 0, 0

    # Iterate over the data loader
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        
        # Apply mask to gradients
        if grad_mask is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_mask:
                    param.grad *= grad_mask[name].to(dtype=param.dtype, device=param.device)
                    
                    
        optimizer.step()

        # Update total loss and correct predictions
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()

    # Calculate average loss and accuracy
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy

def train_steps(model, loader, optimizer, criterion, device, J, grad_mask=None):
    """ Trains the model for a specified number of steps.
    Args:
        model: The model to train.
        loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the training on (CPU or GPU).
        J: Number of training steps to perform.
        grad_mask: Optional mask to apply to gradients.
    Returns:
        Tuple of average loss and accuracy for the steps.
    """
    model.train()
    total_loss, correct = 0, 0
    step = 0
    data_iter = iter(loader)

    # Loop until we reach the specified number of steps
    while step < J:
        # Get the next batch of data, cycling through the loader if needed
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Apply gradient mask if provided
        if grad_mask is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_mask:
                    param.grad *= grad_mask[name].to(dtype=param.dtype, device=param.device)

        optimizer.step()

        # Update total loss and correct predictions
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        step += 1

    # Calculate average loss and accuracy
    avg_loss = total_loss / (step * loader.batch_size)
    accuracy = correct / (step * loader.batch_size)
    return avg_loss, accuracy