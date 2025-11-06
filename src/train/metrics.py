import numpy as np

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))

def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    gains = (ranks < k) / np.log2(ranks + 2)  # +2 so rank 0 -> log2(2)=1
    return float(np.mean(gains))
