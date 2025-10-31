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