import json
import torch
from pathlib import Path

from src.scaling_experiment import (
    run_scaling_experiment, 
    generate_experiments, 
    plot_scaling_comparison,
    plot_runtime_analysis
)


def main():
    with open("config/scaling_config.json", "r") as f:
        cfg = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    experiments = generate_experiments(
        base_batch_size=cfg["base_batch_size"],
        base_muon_lr=cfg["base_muon_lr"],
        base_adamw_lr=cfg["base_adamw_lr"],
        batch_sizes=cfg["batch_sizes"],
        runs=cfg["runs"]
    )
    
    all_results = []
    for idx, (opt_name, bs, muon_lr, adamw_lr, run) in enumerate(experiments, 1):
        print(f"\nExperiment {idx}/{len(experiments)}")
        results = run_scaling_experiment(
            opt_name, bs, muon_lr, adamw_lr, 
            epochs=cfg["epochs"], run=run+1, device=device
        )
        all_results.append(results)
    
    output_file = results_dir / 'scaling_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    plot_scaling_comparison(all_results, results_dir)
    plot_runtime_analysis(all_results, results_dir)
    
    print(f"\nResults: {output_file}")
    print(f"Plots: results/muon_scaling_comparison.png")
    print(f"       results/muon_runtime_analysis.png")


if __name__ == '__main__':
    main()