import torch
import time

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            # Unpack inputs and move to device
            x_cc1, x_cc2, x_mlo1, x_mlo2 = inputs
            x_cc1, x_cc2, x_mlo1, x_mlo2, targets = (
                x_cc1.to(device),
                x_cc2.to(device),
                x_mlo1.to(device),
                x_mlo2.to(device),
                targets.to(device)
            )

            optimizer.zero_grad()
            outputs = model(x_cc1, x_cc2, x_mlo1, x_mlo2)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}, Accuracy: {train_accuracy}%")
