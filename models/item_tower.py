import torch, torch.nn as nn


class ItemTower(nn.Module):
    def __init__(self, n_items: int, d: int=64):
        super().__init__()
        self.emb = nn.Embedding(n_items, d)

    def forward(self, item_ids):
        return self.emb(item_ids)