#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search PatchTST Variants for 10 Synthetic Signals with Evaluation Plots
Usage: python scripts/run_patchtst_grid.py
"""

import os
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from compactformer.patchtst import PatchTSTMinimal, PatchTSTStandard, PatchTSTFull
from compactformer.datasets import PatchDataset
from compactformer.signals import generate_noisy_smooth_signals
from compactformer.train import train_patch_model
from compactformer.eval import (
    evaluate_patch_model,
    evaluate_patch_model_full_aligned,
)
from compactformer.plots import plot_predictions_vs_truth, plot_long_window_predictions

# ----------------------
# Main grid search loop
# ----------------------
def main():

    # -------------------------------
    # Configurable grid search params
    # -------------------------------
    patch_lengths = [4, 8, 12, 16, 20]
    horizons = [4, 8, 12, 16, 20]
    d_models = [8]
    num_heads_list = [2]
    dim_feedforwards = [32]
    num_layers_list = [2]

    epochs = 600
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    param_grid = list(
        itertools.product(
            patch_lengths,
            horizons,
            d_models,
            num_heads_list,
            dim_feedforwards,
            num_layers_list,
        )
    )

    t, signals = generate_noisy_smooth_signals()
    results = []
    preds_dict = {}
    full_preds_dict = {}
    truth_series = {}

    for sig_name, signal in signals.items():
        print(f"\n===== Signal: {sig_name} =====")
        for (
            patch_length,
            horizon,
            d_model,
            num_heads,
            dim_feedforward,
            num_layers,
        ) in param_grid:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(signal.reshape(-1, 1)).squeeze()

            train_data = scaled[:-60]
            test_data = scaled[-(60 + patch_length + horizon) :]

            train_dataset = PatchDataset(train_data, patch_length, horizon)
            test_dataset = PatchDataset(test_data, patch_length, horizon)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            for model_name, ModelClass in [
                ("Minimal", PatchTSTMinimal),
                ("Standard", PatchTSTStandard),
                ("Full", PatchTSTFull),
            ]:
                print(f"Training {model_name} on {sig_name}, patch={patch_length}, horizon={horizon}")
                model = ModelClass(
                    patch_length, horizon, d_model, num_heads, dim_feedforward, num_layers
                ).to(device)
                model, loss_history = train_patch_model(model, train_loader, epochs=epochs)
                rmse, mae, pred_arr, true_arr = evaluate_patch_model(
                    model, test_loader, scaler, horizon
                )

                results.append(
                    {
                        "Signal": sig_name,
                        "Model": model_name,
                        "Patch Length": patch_length,
                        "Horizon": horizon,
                        "d_model": d_model,
                        "Heads": num_heads,
                        "FFN Dim": dim_feedforward,
                        "Layers": num_layers,
                        "RMSE": rmse,
                        "MAE": mae,
                    }
                )

                preds_dict[(sig_name, model_name)] = {"preds": pred_arr, "trues": true_arr}

                start_idx = len(t) - len(test_data)
                full_preds, full_trues = evaluate_patch_model_full_aligned(
                    model,
                    test_loader,
                    scaler,
                    horizon,
                    total_len=len(test_data),
                    start_idx=start_idx,
                )

                if sig_name not in full_preds_dict:
                    full_preds_dict[sig_name] = {}
                full_preds_dict[sig_name][model_name] = full_preds
                if sig_name not in truth_series:
                    truth_series[sig_name] = full_trues

    # ---- Save and Show Results ----
    df_results = pd.DataFrame(results)
    results_csv = os.path.join(results_dir, "noisy_patch_results_full.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    df_results_sorted = df_results.sort_values(by=["Signal", "RMSE"])
    print("\nTop 3 configs per signal and model:")
    print(df_results_sorted.groupby(["Signal", "Model"]).head(3))

    # ---- Plot Predictions ----
    plot_predictions_vs_truth(preds_dict, n_examples=3)

    # ---- Plot Long Window Forecasts ----
    for sig in truth_series.keys():
        plot_long_window_predictions(
            full_preds_dict[sig], truth_series[sig], signal_name=sig
        )

if __name__ == "__main__":
    main()
