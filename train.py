# This script trains a neural network model using PyTorch.

def train(model, loader, optimizer, criterion, device, grad_mask=None):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        
        # apply mask to gradients
        if grad_mask is not None:
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_mask:
                    param.grad *= grad_mask[name].to(param.grad.device)
                    
                    
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy