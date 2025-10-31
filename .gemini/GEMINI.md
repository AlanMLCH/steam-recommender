- prefer functional programming paradigms.
- Stack: fastapi + pytorch + faiss + docker + make + uv python
- try to maintain a similar structure for the project:

game-ml-backend/
├─ configs/
│  ├─ training.yaml
│  └─ service.yaml
├─ data/
│  ├─ raw/             # (gitignored)
│  └─ processed/       # (gitignored)
├─ artifacts/          # (gitignored) faiss index, embeddings, model.pt
├─ models/             # torch checkpoints
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_train.ipynb
│  └─ 03_offline_eval.ipynb
├─ src/
│  ├─ dataio/
│  │  ├─ schema.py
│  │  └─ dataset.py
│  ├─ features/
│  │  └─ featurize.py
│  ├─ modeling/
│  │  ├─ towers.py
│  │  ├─ losses.py
│  │  └─ train.py
│  ├─ index/
│  │  ├─ build_faiss.py
│  │  └─ load.py
│  ├─ service/
│  │  ├─ main.py        # FastAPI app
│  │  └─ inference.py
│  └─ utils/
│     └─ metrics.py
├─ dashboard/
│  └─ app.py            # Streamlit
├─ tests/
│  ├─ test_inference.py
│  └─ test_index.py
├─ docker/
│  ├─ Dockerfile.api
│  ├─ Dockerfile.train
│  └─ Dockerfile.dashboard
├─ docker-compose.yml
├─ Makefile
├─ requirements.txt
└─ README.md


-request is going to be like this:

{
  "user_id": "u_123",
  "time_played": {"g_221": 120.0, "g_501": 8.0},
  "genres": ["rpg", "roguelike", "pixel-art"],
  "achievements": {"g_221": 15, "g_777": 2},
  "top_k": 10
}

-the response should be like this:
{
  "user_id": "u_123",
  "recommendations": [
    {"item_id": "g_742", "score": 0.83, "reason": ["rpg","roguelike"]},
    {"item_id": "g_371", "score": 0.79, "reason": ["pixel-art"]}
  ]
}

CODE:

Minimal training components

src/modeling/towers.py

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

src/modeling/losses.py (InfoNCE / sampled softmax style)
import torch, torch.nn.functional as F

def contrastive_loss(user_emb, item_emb, temperature=0.07):
    logits = (user_emb @ item_emb.T) / temperature
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)


src/modeling/train.py (skeleton)
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


Build the FAISS index
src/index/build_faiss.py

import faiss, numpy as np

def build_index(item_emb: np.ndarray, use_flat=True, pq_m=8, nlist=256):
    d = item_emb.shape[1]
    if use_flat:
        index = faiss.IndexFlatIP(d)  # cosine with normalized vectors
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8)  # 8 bits/code
        index.train(item_emb)
    index.add(item_emb)
    return index

src/index/load.py
import faiss, numpy as np, pathlib, json

def load_index(index_path: str):
    return faiss.read_index(index_path)

def save_index(index, path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)

def topk(index, user_vec: np.ndarray, k: int):
    D, I = index.search(user_vec.astype("float32"), k)
    return I[0], D[0]


FastAPI service
src/service/inference.py

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

src/service/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, json
from .inference import RecommenderRuntime

app = FastAPI(title="Game Recommender API")

class RecommendRequest(BaseModel):
    user_id: str
    time_played: dict = {}
    genres: list[str] = []
    achievements: dict = {}
    top_k: int = 10

# simple bag-of-plays + genres one-hot (placeholder)
def user_vectorize(req: RecommendRequest, genre_vocab: list[str]):
    vec = []
    vec.append(sum(req.time_played.values())/ (1.0 + len(req.time_played)))  # crude density
    vec += [1.0 if g in req.genres else 0.0 for g in genre_vocab]
    return np.array(vec, dtype="float32")

# init runtime (paths from config/env in real code)
GENRE_VOCAB = ["rpg","fps","roguelike","racing","strategy","simulation","sports","adventure","puzzle"]
RUNTIME = RecommenderRuntime("models/model.pt", "artifacts/index.faiss", user_dim=1+len(GENRE_VOCAB))

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    feats = user_vectorize(req, GENRE_VOCAB)
    ids, scores = RUNTIME.recommend(feats, req.top_k)
    # map FAISS ids back to item_ids via artifacts/item_meta.parquet in real code
    return {"user_id": req.user_id, "recommendations": [{"item_id": f"g_{i}", "score": float(s)} for i,s in zip(ids, scores)]}


Streamlit dashboard (quick operator UI, no npm)
dashboard/app.py
import streamlit as st, requests, json

st.title("Game Recommender — Operator Console")
api = st.text_input("API base URL", "http://api:8000")
user_id = st.text_input("User ID", "u_123")
genres = st.multiselect("Genres", ["rpg","fps","roguelike","racing","strategy","simulation","sports","adventure","puzzle"], default=["rpg","roguelike"])
top_k = st.slider("Top-K", 1, 20, 10)

if st.button("Get Recommendations"):
    payload = {"user_id": user_id, "time_played": {"g_221": 120.0}, "genres": genres, "achievements": {}, "top_k": top_k}
    r = requests.post(f"{api}/recommend", json=payload, timeout=30)
    st.json(r.json())


Docker & Compose
requirements.txt
fastapi==0.115.2
uvicorn[standard]==0.30.6
pydantic==2.9.2
torch==2.4.0
faiss-cpu==1.8.0
numpy==2.1.1
pandas==2.2.3
pyarrow==17.0.0
streamlit==1.38.0

docker/Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]

docker/Dockerfile.train
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt && pip install jupyter
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "ipykernel_launcher", "-f", "/dev/null"]

docker/Dockerfile.dashboard
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt
COPY dashboard ./dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

docker-compose.yml
version: "3.9"
services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    volumes:
      - ./artifacts:/app/artifacts
      - ./models:/app/models
  dashboard:
    build: { context: ., dockerfile: docker/Dockerfile.dashboard }
    environment:
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    ports: ["8501:8501"]
    depends_on: [api]

Makefile (nice DX)
.PHONY: build up down train index api test

build:
\tdocker compose build

up:
\tdocker compose up -d

down:
\tdocker compose down

train:
\tdocker run --rm -v $(PWD):/app -w /app --entrypoint python \
\t\t$(shell docker build -q -f docker/Dockerfile.train .) src/modeling/train.py

index:
\tdocker run --rm -v $(PWD):/app -w /app --entrypoint python \
\t\t$(shell docker build -q -f docker/Dockerfile.train .) src/index/build_faiss.py

api:
\tdocker compose up api

test:
\tpytest -q

Notebooks (what to do in each)

01_eda.ipynb: inspect telemetry sparsity, genre coverage, cold-start rates, quick co-play matrix.

02_train.ipynb: build/train the two-tower, export:

models/model.pt

artifacts/item_emb.npy, artifacts/index.faiss, artifacts/item_meta.parquet

03_offline_eval.ipynb: compute Recall@K, NDCG@K, item coverage, genre diversity; save plots to artifacts/plots/.

Basic tests
tests/test_index.py
import numpy as np
from src.index.build_faiss import build_index

def test_faiss_top1_identity():
    X = np.eye(4).astype("float32")
    idx = build_index(X, use_flat=True)
    D, I = idx.search(X, 1)
    assert (I.squeeze() == np.arange(4)).all()

tests/test_inference.py
from fastapi.testclient import TestClient
from src.service.main import app

def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"

Configs (tweak without code changes)
configs/training.yaml (example)
seed: 42
batch_size: 1024
epochs: 2
lr: 0.001
embedding_dim: 64
index:
  type: flat   # flat | ivfpq
  nlist: 256
  pq_m: 8

configs/service.yaml (example)
model_path: models/model.pt
index_path: artifacts/index.faiss
genre_vocab: ["rpg","fps","roguelike","racing","strategy","simulation","sports","adventure","puzzle"]

Run it end-to-end (MVP)

Put a tiny toy catalog in data/processed/items.csv and craft a small synthetic interaction dataset (user_id,item_id).

Run training (from host or make train) to produce models/model.pt + artifacts/item_emb.npy.

Build FAISS index (make index) → creates artifacts/index.faiss.

make build && make up

Test:

curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_1","time_played":{"g_221":120},"genres":["rpg","roguelike"],"achievements":{},"top_k":5}'


Open dashboard: http://localhost:8501