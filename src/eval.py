
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


def evaluate_patch_model(model, loader, scaler, horizon):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            truth = y.numpy().flatten()
            preds.append(pred)
            trues.append(truth)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds.reshape(-1, horizon), trues.reshape(-1, horizon)


def evaluate_patch_model_full(model, loader, scaler, horizon):
    model.eval()
    full_preds, full_trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            truth = y.numpy().flatten()
            full_preds.extend(pred)
            full_trues.extend(truth)

    full_preds = scaler.inverse_transform(np.array(full_preds).reshape(-1, 1)).flatten()
    full_trues = scaler.inverse_transform(np.array(full_trues).reshape(-1, 1)).flatten()

    return full_preds, full_trues


def evaluate_patch_model_full_aligned(
    model, loader, scaler, horizon, total_len, start_idx=0
):
    model.eval()
    sum_preds = np.zeros(start_idx + total_len)
    count_preds = np.zeros(start_idx + total_len)
    sum_truths = np.zeros(start_idx + total_len)
    count_truths = np.zeros(start_idx + total_len)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            preds = model(x).cpu().numpy().flatten()
            truth = y.numpy().flatten()

            for j in range(horizon):
                idx = start_idx + i + j
                if idx < len(sum_preds):
                    sum_preds[idx] += preds[j]
                    count_preds[idx] += 1
                    sum_truths[idx] += truth[j]
                    count_truths[idx] += 1

    avg_preds = np.divide(
        sum_preds, count_preds, out=np.zeros_like(sum_preds), where=count_preds != 0
    )
    avg_truths = np.divide(
        sum_truths, count_truths, out=np.zeros_like(sum_truths), where=count_truths != 0
    )

    avg_preds = scaler.inverse_transform(avg_preds.reshape(-1, 1)).flatten()
    avg_truths = scaler.inverse_transform(avg_truths.reshape(-1, 1)).flatten()

    return avg_preds[start_idx:], avg_truths[start_idx:]






def evaluate_informer_model(model, loader, scaler, horizon):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            true = y.numpy().flatten()
            preds.append(pred)
            trues.append(true)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds.reshape(-1, horizon), trues.reshape(-1, horizon)


def evaluate_informer_long(model, loader, scaler, horizon, total_len, start_idx=0):
    sum_preds = np.zeros(start_idx + total_len)
    count_preds = np.zeros(start_idx + total_len)
    sum_truths = np.zeros(start_idx + total_len)
    count_truths = np.zeros(start_idx + total_len)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            true = y.numpy().flatten()
            for j in range(horizon):
                idx = start_idx + i + j
                if idx < len(sum_preds):
                    sum_preds[idx] += pred[j]
                    count_preds[idx] += 1
                    sum_truths[idx] += true[j]
                    count_truths[idx] += 1
    avg_preds = np.divide(
        sum_preds, count_preds, out=np.zeros_like(sum_preds), where=count_preds != 0
    )
    avg_truths = np.divide(
        sum_truths, count_truths, out=np.zeros_like(sum_truths), where=count_truths != 0
    )
    return (
        scaler.inverse_transform(avg_preds.reshape(-1, 1)).flatten()[start_idx:],
        scaler.inverse_transform(avg_truths.reshape(-1, 1)).flatten()[start_idx:],
    )

####################################


def evaluate_autoformer_model(model, loader, scaler, horizon):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            true = y.numpy().flatten()
            preds.append(pred)
            trues.append(true)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae, preds.reshape(-1, horizon), trues.reshape(-1, horizon)


def evaluate_autoformer_long(model, loader, scaler, horizon, total_len, start_idx=0):
    sum_preds = np.zeros(start_idx + total_len)
    count_preds = np.zeros(start_idx + total_len)
    sum_truths = np.zeros(start_idx + total_len)
    count_truths = np.zeros(start_idx + total_len)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            pred = model(x).cpu().numpy().flatten()
            true = y.numpy().flatten()
            for j in range(horizon):
                idx = start_idx + i + j
                if idx < len(sum_preds):
                    sum_preds[idx] += pred[j]
                    count_preds[idx] += 1
                    sum_truths[idx] += true[j]
                    count_truths[idx] += 1

    avg_preds = np.divide(
        sum_preds, count_preds, out=np.zeros_like(sum_preds), where=count_preds != 0
    )
    avg_truths = np.divide(
        sum_truths, count_truths, out=np.zeros_like(sum_truths), where=count_truths != 0
    )
    return (
        scaler.inverse_transform(avg_preds.reshape(-1, 1)).flatten()[start_idx:],
        scaler.inverse_transform(avg_truths.reshape(-1, 1)).flatten()[start_idx:],
    )
