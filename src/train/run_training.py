import os, json
from pathlib import Path
import pandas as pd
import yaml
from ..data_pipeline.config import data_dir
from .splits import run as run_splits
from .dataset import make_loader
from .trainer import train_loop, evaluate, export_item_embeddings
from .model import TwoTower
from .build_faiss import run as build_faiss_run

def main():
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # 1) splits + counts
    train_df, val_df = run_splits(out_dir=artifacts_dir)
    users_df = pd.read_parquet(artifacts_dir / "users_in_data.parquet")
    items_df = pd.read_parquet(artifacts_dir / "items_in_data.parquet")
    n_users = users_df["user_id"].nunique()
    n_items = items_df["game_id"].nunique()

    # 2) model + loaders
    embed_dim = params["train"]["embed_dim"]
    batch = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    val_k = params["eval"]["val_k"]

    train_loader = make_loader(train_df, batch, shuffle=True)
    model = TwoTower(n_users=n_users, n_items=n_items, dim=embed_dim)

    # 3) train
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    train_loop(model, train_loader, epochs, lr, device)

    # 4) eval
    metrics = evaluate(model, val_df, n_items, val_k, device)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {metrics}")

    # 5) export artifacts
    export_item_embeddings(model, artifacts_dir)
    build_faiss_run(artifacts_dir)

if __name__ == "__main__":
    main()
