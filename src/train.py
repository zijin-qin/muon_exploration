import torch
import time

def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    # Handle both single optimizer and list of optimizers
    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        for opt in optimizer:
            opt.zero_grad()
            
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for opt in optimizer:
            opt.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(trainloader), 100 * correct / total


def test(model, testloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(testloader), 100 * correct / total
