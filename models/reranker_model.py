import torch, torch.nn as nn


class PairwiseMLP(nn.Module):
    def __init__(self, d: int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, u, v):
        x = torch.cat([u, v], dim=-1)
        return self.net(x).squeeze(-1)