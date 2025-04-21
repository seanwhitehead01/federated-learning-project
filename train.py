import torch.nn as nn
from torch.optim import SGD

def train(model, train_loader, device, epochs=10, lr=1e-4):
    # Set the model to train mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    # Move model to GPU if available
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, targets in train_loader:
            # Move data to GPU if available
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_preds / total_preds
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("Training Finished")