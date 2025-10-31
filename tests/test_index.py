import numpy as np
from src.index.build_faiss import build_index

def test_faiss_top1_identity():
    X = np.eye(4).astype("float32")
    idx = build_index(X, use_flat=True)
    D, I = idx.search(X, 1)
    assert (I.squeeze() == np.arange(4)).all()