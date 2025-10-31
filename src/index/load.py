import faiss, numpy as np, pathlib, json

def load_index(index_path: str):
    return faiss.read_index(index_path)

def save_index(index, path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)

def topk(index, user_vec: np.ndarray, k: int):
    D, I = index.search(user_vec.astype("float32"), k)
    return I[0], D[0]