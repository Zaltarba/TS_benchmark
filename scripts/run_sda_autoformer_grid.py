#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search: Autoformer Variants on Noisy Synthetic Signals
Usage: python scripts/run_autoformer_grid.py
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from src.models.sda_autoformer import SDAStandardAutoformer, SDAMinimalAutoformer, SDAFullAutoformer
from src.datasets import TimeSeriesDataset
from src.data_simulation.signals import generate_noisy_smooth_signals
from src.train import train_autoformer_model
from src.eval import evaluate_autoformer_model, evaluate_autoformer_long


# ------------- Main Script -------------- #
def main():

    # -------- Configurable Parameters -------- #
    patch_lengths = [4, 8, 12, 16, 20]
    horizons = [4, 8, 12, 16, 20]
    patch_lengths = [4, 8,]
    horizons = [4, 8,]
    results_dir = "simulation_results_sda"
    os.makedirs(results_dir, exist_ok=True)
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dict = [
        ("Minimal", SDAMinimalAutoformer),
        ("Standard", SDAStandardAutoformer),
        ("Full", SDAFullAutoformer),
    ]

    t, signals = generate_noisy_smooth_signals()
    results = []

    for sig_name, sig in signals.items():
        for patch_length, horizon in itertools.product(patch_lengths, horizons):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sig.reshape(-1, 1)).squeeze()
            train_data = scaled[:-60]
            test_data = scaled[-(60 + patch_length + horizon) :]
            train_ds = TimeSeriesDataset(train_data, patch_length, horizon)
            test_ds = TimeSeriesDataset(test_data, patch_length, horizon)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

            for model_name, ModelClass in models_dict:
                print(
                    f"Training {model_name} | Signal: {sig_name} | Patch: {patch_length}, Horizon: {horizon}"
                )
                model = ModelClass(patch_length, horizon).to(device)
                model = train_autoformer_model(model, train_loader, epochs=epochs)
                rmse, mae, preds, trues = evaluate_autoformer_model(
                    model, test_loader, scaler, horizon
                )
                start_idx = len(t) - len(test_data)
                long_preds, long_trues = evaluate_autoformer_long(
                    model, test_loader, scaler, horizon, len(test_data), start_idx
                )

                results.append(
                    {
                        "Signal": sig_name,
                        "Model": model_name,
                        "Patch Length": patch_length,
                        "Horizon": horizon,
                        "RMSE": rmse,
                        "MAE": mae,
                    }
                )
                if False:
                    # Short Forecast Plot
                    plt.figure(figsize=(12, 3))
                    for i in range(min(3, len(preds))):
                        plt.plot(trues[i], label="True", color="black")
                        plt.plot(preds[i], label="Pred", linestyle="--")
                        plt.title(f"{sig_name} - {model_name} (short)")
                        plt.grid()
                        plt.legend()
                        plt.show()

                    # Long Forecast Plot
                    plt.figure(figsize=(14, 4))
                    plt.plot(t[-len(long_trues) :], long_trues, label="True", color="black")
                    plt.plot(
                        t[-len(long_preds) :], long_preds, label=f"{model_name}", linestyle="--"
                    )
                    plt.title(f"{sig_name} - {model_name} (long)")
                    plt.grid()
                    plt.legend()
                    plt.show()

    # ==== Save and Show Results ====
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, "noisy_autoformer_results_different_signals.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")
    print(
        df_results.sort_values(by=["Signal", "RMSE"]).groupby(["Signal", "Model"]).head(2)
    )

if __name__ == "__main__":
    main()
