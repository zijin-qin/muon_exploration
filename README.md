# Evaluating Muon Optimizer
Quarter 1 Project for 2025-2026 Capstone

## Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/zijin-qin/muon_exploration.git
cd muon_exploration
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

### Experiment 1: AdamW vs Muon Comparison
Compares baseline AdamW optimizer with Muon optimizer using fixed batch size and learning rates.

Run:
```bash
python run.py
```
Outputs:
Training/test metrics printed to console
muon_vs_adamw_cifar10.png - Performance comparison plots

### Experiment 2: Muon Batch Size Analysis
Investigates how Muon's learning rate should scale with batch size, comparing:

Linear scaling: LR ∝ batch_size
Quadratic scaling: LR ∝ batch_size²

Tests batch sizes: [16, 32, 64, 128, 256, 512] with 2 runs each.

Run:
```bash
python run_scaling.py
```
Outputs:
results/scaling_results.json - Raw training metrics
`results/muon_scaling_comparison.png` - Loss/accuracy curves for all batch sizes
`results/muon_runtime_analysis.png` - Runtime analysis plots

Expected runtime: 3-4 hours on a single GPU

## Notes
- Tested on a single GPU. Running on CPU is possible but slower.
- Muon optimizer: https://github.com/KellerJordan/Muon
- Both experiments use the same CNN architecture and training procedures
- CIFAR-10 dataset is automatically downloaded on first run
