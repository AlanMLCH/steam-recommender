import os
from pathlib import Path
import numpy as np
import faiss

def build_flat_index(emb: np.ndarray, out_path: Path):
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine via normalized emb (we'll skip norm for MVP)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, str(out_path))

def run(artifacts_dir: Path):
    emb = np.load(artifacts_dir / "item_emb.npy")
    build_flat_index(emb, artifacts_dir / "index.faiss")
    print(f"[faiss] wrote {artifacts_dir/'index.faiss'}")
