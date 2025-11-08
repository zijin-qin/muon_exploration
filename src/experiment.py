import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from src.model import CIFAR10CNN
from src.data import get_dataloaders
from src.train import train_epoch, test
from muon import SingleDeviceMuon

def run_experiment(optimizer_name, batch_size, muon_lr=0.02, adamw_lr=3e-4, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning experiment {optimizer_name.upper()} with batch size {batch_size}")

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = get_dataloaders(batch_size)

    if 'muon' in optimizer_name:
        from muon import SingleDeviceMuon
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if p.ndim >= 2 and (('conv' in name and 'conv1' not in name and 'weight' in name)
                                or ('fc1' in name and 'weight' in name)):
                muon_params.append(p)
            else:
                adamw_params.append(p)
        muon_opt = SingleDeviceMuon(muon_params, lr=muon_lr, momentum=0.95)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer = [muon_opt, adamw_opt]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=adamw_lr,
                                       betas=(0.9, 0.95), weight_decay=0.01)

    # Scheduler only on first optimizer
    first_opt = optimizer[0] if isinstance(optimizer, list) else optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(first_opt, T_max=epochs)

    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - start_time
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

def plot_results(adamw_results, muon_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(adamw_results['test_acc']) + 1)

    axes[0].plot(epochs, adamw_results['train_loss'], 'o-', label='AdamW', linewidth=2)
    axes[0].plot(epochs, muon_results['train_loss'], 's-', label='Muon', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, adamw_results['test_acc'], 'o-', label='AdamW', linewidth=2)
    axes[1].plot(epochs, muon_results['test_acc'], 's-', label='Muon', linewidth=2)
    axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    times = [sum(adamw_results['epoch_time']), sum(muon_results['epoch_time'])]
    bars = axes[2].bar(['AdamW', 'Muon'], times, color=['#1f77b4', '#ff7f0e'])
    axes[2].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Total Time (seconds)')
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}s', ha='center', va='bottom')
    
    fig.suptitle('Muon vs AdamW on CIFAR-10', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("muon_vs_adamw_cifar10.png", dpi=200)
    plt.close()
    print("Graph saved to muon_vs_adamw_cifar10.png")


def print_summary(results, optimizer_name="Optimizer"):
    print(f"\nSummary for {optimizer_name}:")
    print(f"Final Training Loss: {results['train_loss'][-1]:.4f}")
    print(f"Final Test Loss:     {results['test_loss'][-1]:.4f}")
    print(f"Final Training Acc:  {results['train_acc'][-1]:.2f}%")
    print(f"Final Test Acc:      {results['test_acc'][-1]:.2f}%")
    print(f"Best Test Acc:       {max(results['test_acc']):.2f}%")
    print(f"Total Training Time: {sum(results['epoch_time']):.1f}s")
