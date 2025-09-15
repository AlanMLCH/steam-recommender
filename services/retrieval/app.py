import os, json
from fastapi import FastAPI, Query
import numpy as np
import faiss, torch
from models.user_tower import UserTower


ART = os.getenv("ARTIFACTS_DIR", "/workspace/artifacts")
IDX = os.getenv("INDEX_DIR", "/workspace/index")


USER_EMB_PATH = os.path.join(ART, "user_tower.pt")
ITEM_IDS_PATH = os.path.join(ART, "item_ids.npy")
INDEX_PATH = os.path.join(IDX, "faiss.index")


app = FastAPI(title="Retrieval API")


def load_model():
    model = UserTower(n_users=100000, d=64)
    if os.path.exists(USER_EMB_PATH):
        model.load_state_dict(torch.load(USER_EMB_PATH, map_location="cpu"))
    model.eval()
    return model


MODEL = load_model()


def load_index():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError("FAISS index missing. Run build_index.py or trainer.")
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(ITEM_IDS_PATH)
    return index, ids


FAISS_INDEX, ITEM_IDS = load_index()


@app.get("/retrieve")
def retrieve(user_id: int = Query(...), k: int = Query(50)):
    with torch.no_grad():
        u = torch.tensor([user_id % 100000], dtype=torch.long)
        uemb = MODEL(u).numpy().astype('float32')
    faiss.normalize_L2(uemb)
    scores, idxs = FAISS_INDEX.search(uemb, k)
    item_ids = ITEM_IDS[idxs[0]].tolist()
    return {"candidates": item_ids, "scores": scores[0].tolist()}