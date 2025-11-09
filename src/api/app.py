from pathlib import Path
from typing import List, Optional, Dict
import os
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------- Paths & env ---------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
ARTI_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))

ITEM_EMB_PATH = ARTI_DIR / "item_emb.npy"
FEATURES_PATH = DATA_DIR / "features" / "interactions_features.parquet"
ITEM_LOOKUP_PATH = DATA_DIR / "lookups" / "item_lookup.parquet"

# --------- Models ---------
class RecommendRequest(BaseModel):
    game_ids: Optional[List[int]] = None         # user recent/liked games
    top_k: int = 20
    exclude_seen: bool = True

class RecommendResponseItem(BaseModel):
    game_id: int
    score: float
    title: str

class RecommendResponse(BaseModel):
    items: List[RecommendResponseItem]

# --------- App ---------
app = FastAPI(title="Game Recs API", version="0.1.0")

# Globals populated at startup
INDEX = None         # FAISS index
ITEM_EMB = None      # np.ndarray [n_items, dim]
POPULAR = None       # pd.DataFrame with columns: game_id, score
ITEMS_DF = None      # pd.DataFrame with item metadata
N_ITEMS = 0
DIM = 0

def _build_popularity() -> pd.DataFrame:
    """Simple popularity as total play time (you can switch to #users, purchases, etc.)."""
    if not FEATURES_PATH.exists():
        # Fallback uniform popularity if features not present
        return pd.DataFrame({"game_id": np.arange(N_ITEMS, dtype=np.int32), "score": 1.0})
    df = pd.read_parquet(FEATURES_PATH, columns=["game_id", "play_time_hours"])
    pop = (
        df.groupby("game_id", as_index=False)["play_time_hours"]
          .sum()
          .rename(columns={"play_time_hours": "score"})
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )
    return pop

def _build_index_from_embeddings(emb: np.ndarray) -> faiss.IndexFlatIP:
    """
    For now we use raw inner product. If you prefer cosine:
      - L2-normalize emb rows and also the query vector.
      - Keep IndexFlatIP (IP over normalized vectors == cosine).
    """
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))
    return index

@app.on_event("startup")
def startup():
    global INDEX, ITEM_EMB, POPULAR, ITEMS_DF, N_ITEMS, DIM
    if not ITEM_EMB_PATH.exists():
        raise RuntimeError(f"Missing item embeddings: {ITEM_EMB_PATH}")
    if not ITEM_LOOKUP_PATH.exists():
        raise RuntimeError(f"Missing item lookup: {ITEM_LOOKUP_PATH}")

    ITEM_EMB = np.load(ITEM_EMB_PATH)
    ITEM_EMB = ITEM_EMB / np.linalg.norm(ITEM_EMB, axis=1, keepdims=True)
    N_ITEMS, DIM = ITEM_EMB.shape[0], ITEM_EMB.shape[1]
    
    INDEX = _build_index_from_embeddings(ITEM_EMB)
    POPULAR = _build_popularity()
    ITEMS_DF = pd.read_parquet(ITEM_LOOKUP_PATH)
    
    print(f"[startup] index built with {N_ITEMS} items, dim={DIM}")

def _average_item_embeddings(item_ids: List[int]) -> np.ndarray:
    valid = [i for i in item_ids if 0 <= i < N_ITEMS]
    if not valid:
        return np.zeros((DIM,), dtype=np.float32)
    
    # Give more weight to more recent items
    weights = np.linspace(1.0, 2.0, num=len(valid))
    vecs = ITEM_EMB[np.array(valid, dtype=np.int64)]
    
    # Weighted average
    avg_vec = (vecs * weights[:, np.newaxis]).mean(axis=0)
    return avg_vec.astype(np.float32)

def _faiss_search(query_vec: np.ndarray, k: int, exclude: Optional[List[int]] = None):
    q = query_vec.reshape(1, -1)
    q = q / np.linalg.norm(q)
    D, I = INDEX.search(q, min(k + (len(exclude) if exclude else 0) + 50, N_ITEMS))  # overshoot, we’ll filter
    ids = I[0].tolist()
    scores = D[0].tolist()
    results = []
    ex = set(exclude or [])
    for gid, s in zip(ids, scores):
        if gid in ex:
            continue
        results.append((gid, float(s)))
        if len(results) >= k:
            break
    return results

@app.get("/health")
def health():
    return {"status": "ok", "n_items": N_ITEMS, "dim": DIM}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    top_k = max(1, min(req.top_k, 100))
    exclude = req.game_ids if (req.exclude_seen and req.game_ids) else []

    # Case A: have some recent/liked game_ids → build user vector
    if req.game_ids and len(req.game_ids) > 0:
        u_vec = _average_item_embeddings(req.game_ids)
        if not np.any(u_vec):
            # malformed ids, fall back to popularity
            pop = POPULAR[~POPULAR["game_id"].isin(exclude)].head(top_k)
            recs = pop.merge(ITEMS_DF, on="game_id")
            return RecommendResponse(items=[{"game_id": int(r.game_id), "score": float(r.score), "title": r.title} for r in recs.itertuples()])
        
        results = _faiss_search(u_vec, top_k, exclude)
        recs_df = pd.DataFrame(results, columns=["game_id", "score"])
        recs = recs_df.merge(ITEMS_DF, on="game_id")
        return RecommendResponse(items=[{"game_id": int(r.game_id), "score": float(r.score), "title": r.title} for r in recs.itertuples()])

    # Case B: cold start → popularity
    pop = POPULAR.head(top_k)
    recs = pop.merge(ITEMS_DF, on="game_id")
    return RecommendResponse(items=[{"game_id": int(r.game_id), "score": float(r.score), "title": r.title} for r in recs.itertuples()])
