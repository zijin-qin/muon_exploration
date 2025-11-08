import json
from src.experiment import run_experiment, plot_results, print_summary

def main():
    with open("config/config.json", "r") as f:
        cfg = json.load(f)

    adamw_results = run_experiment('adamw',
                                   batch_size=cfg["batch_size"],
                                   adamw_lr=cfg["adamw_lr"],
                                   epochs=cfg["epochs"])
    muon_results = run_experiment('muon',
                                  batch_size=cfg["batch_size"],
                                  muon_lr=cfg["muon_lr"],
                                  adamw_lr=cfg["adamw_lr"],
                                  epochs=cfg["epochs"])

    plot_results(adamw_results, muon_results)
    print_summary(adamw_results, "AdamW")
    print_summary(muon_results, "Muon")

if __name__ == "__main__":
    main()
