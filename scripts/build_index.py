import os, faiss, numpy as np
ART = os.getenv('ARTIFACTS_DIR','artifacts')
IDX = os.getenv('INDEX_DIR','index')
os.makedirs(IDX, exist_ok=True)
item_embs_path = os.path.join(ART,'item_embs.npy')
item_ids_path = os.path.join(ART,'item_ids.npy')
assert os.path.exists(item_embs_path), 'Missing item_embs.npy. Train first.'
X = np.load(item_embs_path).astype('float32')
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, os.path.join(IDX,'faiss.index'))
np.save(item_ids_path, np.arange(X.shape[0]))
print('FAISS index built.', index.ntotal)