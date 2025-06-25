import torch

def evaluate(model, loader, criterion, device):
    """ Evaluates the model on the given data loader.
    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation data.
        criterion: Loss function to compute the loss.
        device: Device to run the evaluation on (CPU or GPU).
    Returns:
        Tuple of average loss and accuracy for the evaluation.
    """
    model.eval()
    total_loss, correct = 0, 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update total loss and correct predictions
            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()

    # Calculate average loss and accuracy
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy