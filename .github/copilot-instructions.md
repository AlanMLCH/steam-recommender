## Quick orientation

This repo implements a two-tower game recommender with (1) a training pipeline that produces a PyTorch model and item embeddings, (2) a FAISS index for fast nearest-neighbor search, and (3) a FastAPI service + Streamlit dashboard for serving and exploring recommendations.

High-level components (files to inspect):
- Training & models: `src/modeling/*.py` (notably `towers.py`, `losses.py`, `train.py`). The two-tower model produces L2-normalized embeddings (see `MLP` -> forward uses F.normalize).
- Indexing: `src/index/build_faiss.py` (builds flat or IVFPQ indexes) and `src/index/load.py` (read/write/topk).
- Serving: FastAPI app at `src/service/main.py` and runtime wrapper `src/service/inference.py` (loads model and index, exposes `RecommenderRuntime.recommend`).
- Data IO and preprocessing: `scripts/preprocess_data.py`, `scripts/get_data.py`, plus placeholder modules `src/dataio/dataset.py` and `src/features/featurize.py` (currently empty — treat as project-specific extension points).

Config, artifacts and expectations
- Configs: `configs/training.yaml` and `configs/service.yaml` contain canonical paths and hyperparameters (embedding_dim, index type, model_path, index_path, genre_vocab).
- Artifacts: built FAISS index is expected in `artifacts/index.faiss`, item metadata in `artifacts/item_meta.parquet`, and model checkpoints in `models/model.pt`.
- Data: put raw input under `data/raw` and processed outputs under `data/processed` (see `scripts/preprocess_data.py`).

Common workflows / commands
- Build & run services (Docker): `make build` then `make up` (or `docker compose build` / `docker compose up -d`).
- Train: `make train` — this invokes training code in `src/modeling/train.py` which uses `TwoTower` and `contrastive_loss`.
- Build index: `make index` — uses the saved item embeddings and `src/index/build_faiss.py`.
- Tests: run `pytest` from the repo root; tests import FastAPI app via `src.service.main` and construct `TestClient` (see `tests/test_inference.py`).

Key patterns and conventions for the AI helper
- Embedding normalization: embeddings are L2-normalized before being added to FAISS. Use `MLP(...); forward -> F.normalize(...)` as the canonical pattern.
- Index search: The repo uses inner-product indexes (`IndexFlatIP`) to simulate cosine when vectors are normalized. Prefer that pattern when reproducing search behavior (`src/index/build_faiss.py`, `src/index/load.py`).
- TwoTower API: `TwoTower(user_dim, item_dim, emb_dim)` with methods `user_embed(x)` and `item_embed(x)`. Training code expects batches with `batch['user_x']` and `batch['item_x']` (see `src/modeling/train.py`).
- Serving: `RecommenderRuntime` loads a model with `strict=False` (so saved checkpoints may omit unused modules). At serve-time `item_dim` may be 0 and only `user_embed` is used (see `src/service/inference.py`). Keep that when altering load logic.

Examples to reference when writing code
- Recommend request payload (used by `src/service/main.py`):

```json
{
  "user_id": "u_123",
  "time_played": {"g_221": 120.0},
  "genres": ["rpg","roguelike"],
  "achievements": {"g_221": 15},
  "top_k": 10
}
```

- Model embedding dim/hyperparams are set in `configs/training.yaml` (e.g. `embedding_dim: 64`).

Testing and edit guidance
- Tests are lightweight: `tests/test_index.py` validates FAISS behavior with identity vectors; `tests/test_inference.py` checks the FastAPI health endpoint. When you change run-time code, update tests accordingly.
- If you add new CLI scripts, place them under `scripts/` and document make targets in the `Makefile` if they become part of common workflows.

Notes / gotchas discovered in repo
- `src/dataio/dataset.py` and `src/features/featurize.py` are present but empty — they act as extension points. Search `scripts/` and `notebooks/` for the canonical feature generation used by training.
- Model loading defaults to CPU (`map_location='cpu'`) in `RecommenderRuntime`. If you add GPU support, keep backward compatibility by supporting a cpu fallback and `strict=False` when loading checkpoints.
- FAISS index files and models are consumed from `artifacts/` and `models/` paths defined in `configs/service.yaml`. Tests assume these paths (or stubbed equivalents).

If anything here is unclear or you want more detail (examples of batch format for training, the item metadata layout, or Makefile targets), tell me which area to expand and I will iterate.
