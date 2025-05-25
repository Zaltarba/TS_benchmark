
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import square, sawtooth
import math
import itertools


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Plotting Functions ====
def plot_predictions_vs_truth(preds_dict, n_examples=3):
    for (signal, model), data in preds_dict.items():
        preds, trues = data["preds"], data["trues"]
        plt.figure(figsize=(12, 4 * n_examples))
        for i in range(min(n_examples, len(preds))):
            plt.subplot(n_examples, 1, i + 1)
            plt.plot(trues[i], label="True", marker="o")
            plt.plot(preds[i], label="Predicted", marker="x")
            plt.title(f"{signal} - {model} (Example {i+1})")
            plt.xlabel("Horizon Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_long_window_predictions(full_preds_dict, full_truth, signal_name):
    plt.figure(figsize=(14, 5))

    # Determine correct x-axis alignment
    n_total = len(full_truth)
    n_preds = len(list(full_preds_dict.values())[0])  # from one model

    # Align predictions to the correct time index
    start_idx = n_total - n_preds
    x_truth = np.arange(n_total)
    x_preds = np.arange(start_idx, n_total)

    plt.plot(x_truth, full_truth, label="Ground Truth", color="black", linewidth=2)
    for model_name, preds in full_preds_dict.items():
        plt.plot(x_preds, preds, label=f"{model_name} Prediction", linestyle="--")

    plt.title(f"Long Window Forecast - {signal_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
