import torch, torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, n_users: int, d: int=64):
        super().__init__()
        self.emb = nn.Embedding(n_users, d)
    def forward(self, user_ids):
        return self.emb(user_ids)