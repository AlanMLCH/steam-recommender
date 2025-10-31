import torch, numpy as np
from ..index.load import load_index, topk
from ..modeling.towers import TwoTower

class RecommenderRuntime:
    def __init__(self, model_path, index_path, user_dim):
        self.device = "cpu"
        self.model = TwoTower(user_dim=user_dim, item_dim=0, emb_dim=64)  # item tower not needed at serve
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()
        self.index = load_index(index_path)

    def encode_user(self, feats: np.ndarray):
        with torch.no_grad():
            u = self.model.user_embed(torch.tensor(feats, dtype=torch.float32, device=self.device))
        return u.cpu().numpy()

    def recommend(self, user_feats: np.ndarray, k: int = 10):
        u = self.encode_user(user_feats[None, :])
        ids, scores = topk(self.index, u, k)
        return ids.tolist(), scores.tolist()