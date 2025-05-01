# This script trains a neural network model using PyTorch.

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy