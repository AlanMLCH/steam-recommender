import torch, torch.optim as optim
from .towers import TwoTower
from .losses import contrastive_loss

def train_loop(dataloader, user_dim, item_dim, epochs=2, lr=1e-3, device="cpu"):
    model = TwoTower(user_dim, item_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            user_x, item_x = batch["user_x"].to(device), batch["item_x"].to(device)
            u = model.user_embed(user_x)
            v = model.item_embed(item_x)
            loss = contrastive_loss(u, v)
            opt.zero_grad(); loss.backward(); opt.step()
    return model