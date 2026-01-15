import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import square, sawtooth

import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)



# ==== Smooth Signal Generator ====


def normalize_to_unit(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def generate_smooth_signals(n=500):
    t = np.arange(n)
    return t, {
        "Sine": normalize_to_unit(np.sin(2 * np.pi * t / 40)),
        "Cosine_Trend": normalize_to_unit(0.01 * t + np.cos(2 * np.pi * t / 50)),
        "ExpDecaySine": normalize_to_unit(
            np.exp(-0.01 * t) * np.sin(2 * np.pi * t / 50)
        ),
        "Poly": normalize_to_unit(0.0001 * t**2 - 0.03 * t + 3),
        "LogSine": normalize_to_unit(np.log1p(t) * np.sin(2 * np.pi * t / 80)),
        "Gaussian": normalize_to_unit(np.exp(-((t - 250) ** 2) / (2 * 50**2))),
        "LongSine": normalize_to_unit(np.sin(2 * np.pi * t / 100)),
        "Cubic": normalize_to_unit(0.00001 * (t - 250) ** 3 + 0.05 * t),
        "Exponential": normalize_to_unit(np.exp(0.005 * t)),
        "CosEnvelope": normalize_to_unit(
            (1 + 0.5 * np.cos(2 * np.pi * t / 100)) * np.sin(2 * np.pi * t / 30)
        ),
    }




def normalize_to_unit(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def add_realistic_noise(signal, std=0.08, amp_jitter=0.05, shift_prob=0.1):
    noisy = signal.copy()

    # Add Gaussian noise
    noisy += np.random.normal(0, std, size=noisy.shape)

    # Add slight amplitude jitter
    jitter_factor = 1 + np.random.normal(0, amp_jitter)
    noisy *= jitter_factor

    # Time shift with wrap-around
    if np.random.rand() < shift_prob:
        shift = np.random.randint(-10, 10)
        noisy = np.roll(noisy, shift)

    return normalize_to_unit(noisy)


def generate_noisy_smooth_signals(n=500):
    t = np.arange(n)
    clean_signals = {
        "Sine": np.sin(2 * np.pi * t / 40),
        "Cosine + Trend": 0.01 * t + np.cos(2 * np.pi * t / 50),
        "Exp. Decay × Sine": np.exp(-0.01 * t) * np.sin(2 * np.pi * t / 50),
        "2nd Order Polynomial": 0.0001 * t**2 - 0.03 * t + 3,
        "Log × Sine": np.log1p(t) * np.sin(2 * np.pi * t / 80),
        "Gaussian Bump": np.exp(-((t - 250) ** 2) / (2 * 50**2)),
        "Long Period Sine": np.sin(2 * np.pi * t / 100),
        "Cubic Polynomial": 0.00001 * (t - 250) ** 3 + 0.05 * t,
        "Exponential Growth": np.exp(0.005 * t),
        "Cosine Envelope × Sine": (1 + 0.5 * np.cos(2 * np.pi * t / 100))
        * np.sin(2 * np.pi * t / 30),
    }

    # Add noise and normalize
    noisy_signals = {
        name: normalize_to_unit(add_realistic_noise(sig))
        for name, sig in clean_signals.items()
    }
    return t, noisy_signals
