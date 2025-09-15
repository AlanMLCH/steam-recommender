import os, json, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.user_tower import UserTower
from models.item_tower import ItemTower


ART = os.getenv('ARTIFACTS_DIR','artifacts')
DATA = os.getenv('DATA_DIR','data')
D = 64


class TripletDataset(Dataset):
    def __init__(self, n_users=10000, n_items=5000, n_samples=200000):
        rng = np.random.default_rng(42)
        self.users = rng.integers(0, n_users, size=(n_samples,))
        self.items = rng.integers(0, n_items, size=(n_samples,))
        self.n_users = n_users
        self.n_items = n_items
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return int(self.users[idx]), int(self.items[idx])


def inbatch_ce(u, v):
    u = nn.functional.normalize(u, dim=-1)
    v = nn.functional.normalize(v, dim=-1)
    logits = u @ v.t() # [B, B]
    target = torch.arange(u.size(0), device=u.device)
    loss = nn.functional.cross_entropy(logits, target)
    return loss


def train():
        os.makedirs(ART, exist_ok=True)
        ds = TripletDataset()
        dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
        ut = UserTower(ds.n_users, D)
        it = ItemTower(ds.n_items, D)
        opt = torch.optim.Adam(list(ut.parameters()) + list(it.parameters()), lr=1e-2)
        for epoch in range(3):
            for b,(u,i) in enumerate(dl):
                uemb = ut(u)
                iemb = it(i)
                loss = inbatch_ce(uemb, iemb)
                opt.zero_grad(); loss.backward(); opt.step()
                if b % 100 == 0:
                    print(f"epoch {epoch} batch {b} loss {loss.item():.4f}")
        # save towers
        torch.save(ut.state_dict(), os.path.join(ART, 'user_tower.pt'))
        torch.save(it.state_dict(), os.path.join(ART, 'item_tower.pt'))
        # export item embeddings and ids (for FAISS)
        with torch.no_grad():
            all_items = torch.arange(ds.n_items)
            item_embs = it(all_items).cpu().numpy().astype('float32')
            np.save(os.path.join(ART,'item_embs.npy'), item_embs)
            np.save(os.path.join(ART,'item_ids.npy'), np.arange(ds.n_items))
        # export user emb snapshot for TorchServe demo
        with torch.no_grad():
            all_users = torch.arange(ds.n_users)
            user_embs = ut(all_users).cpu().numpy().astype('float32')
            np.save(os.path.join(ART,'user_embs.npy'), user_embs)
            # baseline stats for PSI (toy)
            baseline = {"user_norm_mean": float(np.linalg.norm(user_embs, axis=1).mean())}
        with open(os.path.join(ART,'baseline.json'),'w') as f:
            json.dump(baseline, f)


if __name__ == "__main__":
    train()