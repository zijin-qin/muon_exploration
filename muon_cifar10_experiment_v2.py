# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# %%
!pip install git+https://github.com/KellerJordan/Muon
from muon import SingleDeviceMuonWithAuxAdam

# %%
# CNN architecture
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten and FC layers
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# %%
# Data loading
def get_dataloaders(batch_size):
    """
    Returns CIFAR-10 train/test loaders with standard augmentation.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=2)

    return trainloader, testloader

# %%
def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(trainloader)
    acc = 100 * correct / total
    return avg_loss, acc


def test(model, testloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(testloader)
    acc = 100 * correct / total
    return avg_loss, acc

# %%
def run_experiment(optimizer_name, batch_size, muon_lr=0.02, adamw_lr=3e-4, epochs=20):
    """
    Run single experiment with specified optimizer and batch size.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment {optimizer_name.upper()} with batch size {batch_size}")
    
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    trainloader, testloader = get_dataloaders(batch_size)
    
    # Optimizer setup
    if 'muon' in optimizer_name:
        from muon import SingleDeviceMuon

        muon_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            # Muon only for 2D weights
            if p.ndim >= 2 and (
                ('conv' in name and 'conv1' not in name and 'weight' in name)
                or ('fc1' in name and 'weight' in name)
            ):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        muon_opt = SingleDeviceMuon(muon_params, lr=muon_lr, momentum=0.95)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer = [muon_opt, adamw_opt]
        is_muon = True
    else:
        optimizer = [torch.optim.AdamW(model.parameters(), lr=adamw_lr, 
                                       betas=(0.9, 0.95), weight_decay=0.01)]
        is_muon = False
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=epochs)
    
    # Training loop
    results = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        for opt in optimizer:
            opt.zero_grad()

        # Train
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer[0], criterion, device
        )

        for opt in optimizer:
            opt.step()

        # Test
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Store results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    return results

# %%
adamw_results = run_experiment('adamw', batch_size=128, adamw_lr=3e-4, epochs=20)
muon_results = run_experiment('muon', batch_size=128, muon_lr=0.02, adamw_lr=3e-4, epochs=20)

# %%
def plot_results(adamw_results, muon_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(adamw_results['test_acc']) + 1)

    # Training Loss
    axes[0].plot(epochs, adamw_results['train_loss'], 'o-', label='AdamW', linewidth=2)
    axes[0].plot(epochs, muon_results['train_loss'], 's-', label='Muon', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test Accuracy
    axes[1].plot(epochs, adamw_results['test_acc'], 'o-', label='AdamW', linewidth=2)
    axes[1].plot(epochs, muon_results['test_acc'], 's-', label='Muon', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Total training time comparison
    times = [sum(adamw_results['epoch_time']), sum(muon_results['epoch_time'])]
    bars = axes[2].bar(['AdamW', 'Muon'], times, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    axes[2].set_ylabel('Total Time (seconds)', fontsize=12)
    axes[2].set_title('Training Time', fontsize=14, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}s',
                     ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Muon vs AdamW on CIFAR-10', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_results(results_adamw, results_muon)

# %%
def print_summary(results, optimizer_name="Optimizer"):
    """Print final metrics for quick analysis"""
    final_train_loss = results['train_loss'][-1]
    final_test_loss = results['test_loss'][-1]
    final_train_acc = results['train_acc'][-1]
    final_test_acc = results['test_acc'][-1]
    total_time = sum(results['epoch_time'])
    best_test_acc = max(results['test_acc'])
    
    print(f"\nSummary for {optimizer_name}:")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Test Loss:     {final_test_loss:.4f}")
    print(f"Final Training Acc:  {final_train_acc:.2f}%")
    print(f"Final Test Acc:      {final_test_acc:.2f}%")
    print(f"Best Test Acc:       {best_test_acc:.2f}%")
    print(f"Total Training Time: {total_time:.1f}s")

print_summary(adamw_results, "AdamW")
print_summary(muon_results, "Muon")

# %%



