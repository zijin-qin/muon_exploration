"""
Learning rate scaling experiment for Muon optimizer
Compares linear vs quadratic scaling across different batch sizes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.model import CIFAR10CNN
from src.data import get_dataloaders
from src.train import train_epoch, test
from muon import SingleDeviceMuon


def run_scaling_experiment(optimizer_name, batch_size, muon_lr, adamw_lr, 
                           epochs=20, run=1, device='cuda'):
    """Run single scaling experiment"""
    print(f"\n>>> Experiment: {optimizer_name.upper()} | BS={batch_size} | Run={run}")
    print(f"LR: Muon={muon_lr}, AdamW={adamw_lr}")
    
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    trainloader, testloader = get_dataloaders(batch_size)
    
    # Setup Muon + AdamW optimizers
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if p.ndim >= 2 and (
            ('conv' in name and 'conv1' not in name and 'weight' in name)
            or ('fc1' in name and 'weight' in name)
        ):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    print(f"Parameter split: Muon={len(muon_params)}, AdamW={len(adamw_params)}")

    muon_opt = SingleDeviceMuon(muon_params, lr=muon_lr, momentum=0.95)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, 
                                 betas=(0.9, 0.95), weight_decay=0.01)
    optimizer = [muon_opt, adamw_opt]
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=epochs)
    
    results = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': [],
        'optimizer': optimizer_name,
        'batch_size': batch_size,
        'muon_lr': muon_lr,
        'adamw_lr': adamw_lr,
        'run': run
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        for opt in optimizer:
            opt.zero_grad()

        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer[0], criterion, device
        )

        for opt in optimizer:
            opt.step()

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


def generate_experiments(base_batch_size, base_muon_lr, base_adamw_lr, 
                        batch_sizes, runs):
    """Generate experiment configurations"""
    experiments = []
    
    for bs in batch_sizes:
        scale = bs / base_batch_size
        for run in range(runs):
            # Linear scaling
            experiments.append((
                'muon_linear', bs, 
                base_muon_lr * scale, 
                base_adamw_lr * scale, 
                run
            ))
            # Quadratic scaling
            experiments.append((
                'muon_quadratic', bs,
                base_muon_lr * (scale ** 2),
                base_adamw_lr * (scale ** 2),
                run
            ))
    
    return experiments


def plot_scaling_comparison(all_results, output_dir, train_size=50000):
    """
    Plot training loss and test accuracy vs total samples processed.
    """
    linear_results = [r for r in all_results if 'linear' in r['optimizer']]
    quadratic_results = [r for r in all_results if 'quadratic' in r['optimizer']]
    
    # Colors for different batch sizes
    batch_sizes = sorted(set(r['batch_size'] for r in all_results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))
    batch_color_map = {bs: colors[i] for i, bs in enumerate(batch_sizes)}
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Linear scaling: Training loss
    ax = axes[0, 0]
    for res in linear_results:
        batch_size = res['batch_size']
        epochs = np.arange(1, len(res['train_loss']) + 1)
        total_samples = epochs * train_size
        color = batch_color_map[batch_size]
        ax.plot(total_samples, res['train_loss'], 
                color=color, alpha=0.8, linewidth=2,
                label=f"BS{batch_size} (run{res['run']})")
    ax.set_xlabel('Total Samples Processed', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Linear Scaling: Training Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    
    # Linear scaling: Test accuracy
    ax = axes[0, 1]
    for res in linear_results:
        batch_size = res['batch_size']
        epochs = np.arange(1, len(res['test_acc']) + 1)
        total_samples = epochs * train_size
        color = batch_color_map[batch_size]
        ax.plot(total_samples, res['test_acc'], 
                color=color, alpha=0.8, linewidth=2,
                label=f"BS{batch_size} (run{res['run']})")
    ax.set_xlabel('Total Samples Processed', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Linear Scaling: Test Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    
    # Quadratic scaling: Training loss
    ax = axes[1, 0]
    for res in quadratic_results:
        batch_size = res['batch_size']
        epochs = np.arange(1, len(res['train_loss']) + 1)
        total_samples = epochs * train_size
        color = batch_color_map[batch_size]
        ax.plot(total_samples, res['train_loss'], 
                color=color, alpha=0.8, linewidth=2,
                label=f"BS{batch_size} (run{res['run']})")
    ax.set_xlabel('Total Samples Processed', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Quadratic Scaling: Training Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    
    # Quadratic scaling: Test accuracy
    ax = axes[1, 1]
    for res in quadratic_results:
        batch_size = res['batch_size']
        epochs = np.arange(1, len(res['test_acc']) + 1)
        total_samples = epochs * train_size
        color = batch_color_map[batch_size]
        ax.plot(total_samples, res['test_acc'], 
                color=color, alpha=0.8, linewidth=2,
                label=f"BS{batch_size} (run{res['run']})")
    ax.set_xlabel('Total Samples Processed', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Quadratic Scaling: Test Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    
    # Overall title
    fig.suptitle('Muon Optimizer: Linear vs Quadratic Scaling Comparison', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'muon_scaling_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.close()


def plot_runtime_analysis(all_results, output_dir):
    """Plot runtime metrics vs batch size"""
    linear_results = [r for r in all_results if 'linear' in r['optimizer']]
    quadratic_results = [r for r in all_results if 'quadratic' in r['optimizer']]
    
    linear_time = defaultdict(list)
    quadratic_time = defaultdict(list)
    
    for res in linear_results:
        bs = res['batch_size']
        linear_time[bs].append({
            'epoch_time': np.mean(res['epoch_time']),
            'total_time': np.sum(res['epoch_time'])
        })
    
    for res in quadratic_results:
        bs = res['batch_size']
        quadratic_time[bs].append({
            'epoch_time': np.mean(res['epoch_time']),
            'total_time': np.sum(res['epoch_time'])
        })
    
    batch_sizes = sorted(linear_time.keys())
    
    linear_epoch_mean = [np.mean([r['epoch_time'] for r in linear_time[bs]]) 
                        for bs in batch_sizes]
    linear_total_mean = [np.mean([r['total_time'] for r in linear_time[bs]]) 
                        for bs in batch_sizes]
    quadratic_epoch_mean = [np.mean([r['epoch_time'] for r in quadratic_time[bs]]) 
                           for bs in batch_sizes]
    quadratic_total_mean = [np.mean([r['total_time'] for r in quadratic_time[bs]]) 
                           for bs in batch_sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time per epoch
    ax = axes[0]
    ax.plot(batch_sizes, linear_epoch_mean, 'o-', linewidth=2, markersize=8,
            label='Linear Scaling', color='tab:blue')
    ax.plot(batch_sizes, quadratic_epoch_mean, 's--', linewidth=2, markersize=8,
            label='Quadratic Scaling', color='tab:orange')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Average Time per Epoch (s)', fontsize=12)
    ax.set_title('Training Speed: Time per Epoch', fontsize=13, fontweight='bold')
    ax.set_xticks(batch_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Total training time
    ax = axes[1]
    ax.plot(batch_sizes, linear_total_mean, 'o-', linewidth=2, markersize=8,
            label='Linear Scaling', color='tab:blue')
    ax.plot(batch_sizes, quadratic_total_mean, 's--', linewidth=2, markersize=8,
            label='Quadratic Scaling', color='tab:orange')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Total Training Time (s)', fontsize=12)
    ax.set_title('Total Training Cost', fontsize=13, fontweight='bold')
    ax.set_xticks(batch_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'muon_runtime_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    plt.close()