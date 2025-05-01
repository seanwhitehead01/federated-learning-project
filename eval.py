# This script evaluates a PyTorch model on a given dataset.

import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy