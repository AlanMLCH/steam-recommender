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
