What you’ll add

New folders & files

src/
  train/
    splits.py
    dataset.py
    model.py
    metrics.py
    trainer.py
    build_faiss.py
    run_training.py
artifacts/
  (created at runtime: item_emb.npy, index.faiss, metrics.json)


New deps (append to requirements.txt)

torch>=2.3.0
faiss-cpu==1.8.0
scikit-learn>=1.4.0
tqdm>=4.66.0


docker-compose.yml (add a new service)

services:
  pipeline:
    build: .
    image: game-rec-stage1:latest
    environment:
      DATA_DIR: /app/data
    volumes:
      - ./data:/app/data
    command: ["python", "src/pipeline.py", "--step", "all"]

  trainer:
    build: .
    image: game-rec-stage1:latest
    environment:
      DATA_DIR: /app/data
      ARTIFACTS_DIR: /app/artifacts
      TRAIN_EPOCHS: "5"
      BATCH_SIZE: "1024"
      EMBED_DIM: "64"
      LR: "0.001"
      VAL_K: "20"
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
    command: ["python", "src/train/run_training.py"]
    depends_on:
      - pipeline


You can tweak TRAIN_EPOCHS, BATCH_SIZE, etc. via env vars.

1) Split users into train/val (leave-one-out)

src/train/splits.py

from pathlib import Path
import os
import numpy as np
import pandas as pd
from typing import Tuple
from ..config import path_interactions_features

def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)

def load_data() -> pd.DataFrame:
    return pd.read_parquet(path_interactions_features())

def _last_item_per_user(df: pd.DataFrame) -> pd.DataFrame:
    # use max play_time_hours as proxy for "most engaged" item
    # (you can replace with timestamp once you have it)
    df = df.sort_values(["user_id", "play_time_hours"], ascending=[True, False])
    last = df.groupby("user_id", as_index=False).first()[["user_id", "game_id"]]
    last = last.rename(columns={"game_id": "val_item"})
    return last

def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_pairs = _last_item_per_user(df)
    df_merged = df.merge(val_pairs, on="user_id", how="left")
    train = df_merged[df_merged["game_id"] != df_merged["val_item"]].copy()
    val = df_merged.drop_duplicates(["user_id"])[["user_id", "val_item"]].copy()
    val = val.rename(columns={"val_item": "game_id"})
    return train, val

def save_mappings(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # item count & user count help build embedding tables
    df[["user_id"]].drop_duplicates().to_parquet(out_dir / "users_in_data.parquet", index=False)
    df[["game_id"]].drop_duplicates().to_parquet(out_dir / "items_in_data.parquet", index=False)

def run(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    train, val = train_val_split(data)
    save_mappings(data, out_dir)
    return train[["user_id", "game_id"]], val[["user_id", "game_id"]]

2) Dataset & DataLoader (in-batch negatives)

src/train/dataset.py

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

3) Two-tower model (ID embeddings + dot-product)

src/train/model.py

import torch
import torch.nn as nn

class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, u, pos_i):
        u_vec = self.user_emb(u)          # [B, D]
        pos_vec = self.item_emb(pos_i)    # [B, D]
        # in-batch negatives: logits = u @ items_in_batch^T
        # items_in_batch are pos_vec from the same batch
        logits = torch.matmul(u_vec, pos_vec.t())  # [B, B]
        return logits

    def user_encoding(self, u):
        return self.user_emb(u)

    def item_encoding_all(self):
        return self.item_emb.weight.detach()

4) Metrics: Recall@K & NDCG@K (user-wise)

src/train/metrics.py

import numpy as np

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))

def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    gains = (ranks < k) / np.log2(ranks + 2)  # +2 so rank 0 -> log2(2)=1
    return float(np.mean(gains))

5) Trainer (InfoNCE with in-batch negatives)

src/train/trainer.py

import os, json
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import recall_at_k, ndcg_at_k
from .model import TwoTower

def _env_int(name, default):
    return int(os.environ.get(name, str(default)))

def _env_float(name, default):
    return float(os.environ.get(name, str(default)))

def _get_counts(users_df, items_df):
    n_users = users_df["user_id"].nunique()
    n_items = items_df["game_id"].nunique()
    return n_users, n_items

def train_loop(model: TwoTower, train_loader: DataLoader, epochs: int, lr: float, device: str):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # InfoNCE: labels are 0..B-1 (diagonal)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for u, pos in tqdm(train_loader, desc=f"Epoch {ep}"):
            u = u.to(device)
            pos = pos.to(device)
            logits = model(u, pos)                # [B, B]
            target = torch.arange(len(u), device=device)  # diagonal are positives
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[train] epoch={ep} loss={running / len(train_loader):.4f}")

def evaluate(model: TwoTower, val_df, all_items, k: int, device: str):
    model.eval()
    device = torch.device(device)
    with torch.no_grad():
        item_mat = model.item_encoding_all().to(device)  # [I, D]
        ranks = []
        for _, row in val_df.iterrows():
            u = torch.tensor([row["user_id"]], dtype=torch.long, device=device)
            gi = int(row["game_id"])
            u_vec = model.user_encoding(u)               # [1, D]
            sims = torch.matmul(u_vec, item_mat.t()).squeeze(0)  # [I]
            # rank of the true item among all items (descending similarity)
            order = torch.argsort(sims, descending=True)
            rank = (order == gi).nonzero(as_tuple=False)
            if rank.numel() == 0:
                continue
            ranks.append(int(rank.item()))
        ranks = np.array(ranks)
        return {
            f"recall@{k}": recall_at_k(ranks, k),
            f"ndcg@{k}": ndcg_at_k(ranks, k),
            "n_eval_users": int(len(ranks)),
        }

def export_item_embeddings(model: TwoTower, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = model.item_encoding_all().cpu().numpy()
    np.save(out_dir / "item_emb.npy", emb)

6) Build FAISS index

src/train/build_faiss.py

import os
from pathlib import Path
import numpy as np
import faiss

def build_flat_index(emb: np.ndarray, out_path: Path):
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine via normalized emb (we'll skip norm for MVP)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, str(out_path))

def run(artifacts_dir: Path):
    emb = np.load(artifacts_dir / "item_emb.npy")
    build_flat_index(emb, artifacts_dir / "index.faiss")
    print(f"[faiss] wrote {artifacts_dir/'index.faiss'}")

7) Orchestration entrypoint

src/train/run_training.py

import os, json
from pathlib import Path
import pandas as pd
from ..config import data_dir
from .splits import run as run_splits
from .dataset import make_loader
from .trainer import _env_float, _env_int, train_loop, evaluate, export_item_embeddings
from .model import TwoTower
from .build_faiss import run as build_faiss_run

def main():
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) splits + counts
    train_df, val_df = run_splits(out_dir=artifacts_dir)
    users_df = pd.read_parquet(artifacts_dir / "users_in_data.parquet")
    items_df = pd.read_parquet(artifacts_dir / "items_in_data.parquet")
    n_users = users_df["user_id"].nunique()
    n_items = items_df["game_id"].nunique()

    # 2) model + loaders
    embed_dim = _env_int("EMBED_DIM", 64)
    batch = _env_int("BATCH_SIZE", 1024)
    epochs = _env_int("TRAIN_EPOCHS", 5)
    lr = _env_float("LR", 1e-3)
    val_k = _env_int("VAL_K", 20)

    train_loader = make_loader(train_df, batch, shuffle=True)
    model = TwoTower(n_users=n_users, n_items=n_items, dim=embed_dim)

    # 3) train
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    train_loop(model, train_loader, epochs, lr, device)

    # 4) eval
    metrics = evaluate(model, val_df, n_items, val_k, device)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {metrics}")

    # 5) export artifacts
    export_item_embeddings(model, artifacts_dir)
    build_faiss_run(artifacts_dir)

if __name__ == "__main__":
    main()