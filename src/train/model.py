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
