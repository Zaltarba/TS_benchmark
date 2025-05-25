import torch
from torch.utils.data import Dataset



# ==== Dataset Class ====
class PatchDataset(Dataset):
    def __init__(self, data, patch_length, horizon):
        self.X, self.y = self.create_patches(data, patch_length, horizon)

    def create_patches(self, data, patch_length, horizon):
        X, y = [], []
        for i in range(len(data) - patch_length - horizon):
            X.append(data[i : i + patch_length])
            y.append(data[i + patch_length : i + patch_length + horizon])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    



# ==== Dataset ====
class TimeSeriesDataset(Dataset):
    def __init__(self, data, patch_length, horizon):
        self.X, self.y = self.create_sequences(data, patch_length, horizon)

    def create_sequences(self, data, patch_length, horizon):
        X, y = [], []
        for i in range(len(data) - patch_length - horizon):
            X.append(data[i : i + patch_length])
            y.append(data[i + patch_length : i + patch_length + horizon])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
