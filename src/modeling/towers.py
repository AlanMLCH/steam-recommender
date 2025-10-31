import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [nn.ReLU(), nn.Dropout(0.1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, emb_dim=64):
        super().__init__()
        self.user_tower = MLP([user_dim, 128, emb_dim])
        self.item_tower = MLP([item_dim, 128, emb_dim])

    def user_embed(self, x): return self.user_tower(x)
    def item_embed(self, x): return self.item_tower(x)