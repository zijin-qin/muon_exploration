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

## Running the Code
To run the training experiments:
```bash
python run.py
```
- The code automatically downloads the CIFAR-10 dataset.
- Trains the CNN with both AdamW and Muon optimizers.
- Prints training/test loss and accuracy per epoch.
- Plots training results (loss, accuracy, training time comparison).

## Notes
- Tested on a single GPU. Running on CPU is possible but slower.
- Muon optimizer used is implemented by https://github.com/KellerJordan/Muon. 
- Training epochs are set to 20 by default; you can change this in project1_train.py.