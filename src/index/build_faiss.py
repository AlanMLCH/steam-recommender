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