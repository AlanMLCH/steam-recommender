import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.as_tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.as_tensor(df["game_id"].values, dtype=torch.long)

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx]

def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    ds = PairDataset(df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
