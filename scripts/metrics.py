import numpy as np


def recall_at_k(y_true, y_pred, k=10):
    hits = 0
    for t, p in zip(y_true, y_pred):
        hits += int(t in set(p[:k]))
    return hits / len(y_true)


def ndcg_at_k(y_true, y_pred, k=10):
    def dcg(rel):
        return sum((r / np.log2(i+2)) for i, r in enumerate(rel))
    scores = []
    for t, p in zip(y_true, y_pred):
        rel = [1 if x==t else 0 for x in p[:k]]
        ideal = [1] + [0]*(k-1)
        scores.append(dcg(rel) / dcg(ideal))
    return float(np.mean(scores))