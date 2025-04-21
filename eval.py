import torch
import torch.nn as nn

def eval(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct_preds = 0
    total_preds = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to GPU if available
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct_preds / total_preds
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")