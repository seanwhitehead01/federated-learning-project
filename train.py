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

def train_steps(model, loader, optimizer, criterion, device, J, grad_mask=None):
    model.train()
    total_loss, correct = 0, 0
    step = 0
    data_iter = iter(loader)

    while step < J:
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
                    param.grad *= grad_mask[name].to(param.grad.device)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        step += 1

    avg_loss = total_loss / (step * loader.batch_size)
    accuracy = correct / (step * loader.batch_size)
    return avg_loss, accuracy