import os, json
import numpy as np


ART = os.getenv("ARTIFACTS_DIR", "/workspace/artifacts")


class Reranker:
    def initialize(self, ctx):
        # Load precomputed embeddings for demo
        self.user_embs = np.load(os.path.join(ART, 'user_embs.npy')) if os.path.exists(os.path.join(ART,'user_embs.npy')) else None
        self.item_embs = np.load(os.path.join(ART, 'item_embs.npy')) if os.path.exists(os.path.join(ART,'item_embs.npy')) else None
        self.bias = 0.05 # champion bias


    def preprocess(self, data):
        body = data[0].get('body')
        if isinstance(body, (bytes, bytearray)):
            body = body.decode('utf-8')
        payload = json.loads(body)
        return payload


    def inference(self, inputs):
        user_id = inputs['user_id']
        cands = inputs['candidates']
        d = 64
        rng = np.random.default_rng(user_id)
        u = rng.normal(size=(d,)) if self.user_embs is None else self.user_embs[user_id % len(self.user_embs)]
        scores = []
        for cid in cands:
            v = rng.normal(size=(d,)) if self.item_embs is None else self.item_embs[cid % len(self.item_embs)]
            s = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8) + self.bias)
            scores.append((cid, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scores]


    def postprocess(self, preds):
        return [json.dumps({"ranked_ids": preds})]


_service = Reranker()


def handle(data, context):
    if not hasattr(_service, 'initialized'):
        _service.initialize(context)
        _service.initialized = True
    payload = _service.preprocess(data)
    ranked = _service.inference(payload)
    return _service.postprocess(ranked)