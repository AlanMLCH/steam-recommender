import pandas as pd
import numpy as np
from ..data_pipeline.config import path_interactions, path_interactions_features, dir_features

def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(path_interactions())

def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Safe transforms for skewed playtime
    out["play_time_hours_log1p"] = np.log1p(out["play_time_hours"])
    out["play_time_hours_sqrt"] = np.sqrt(out["play_time_hours"])

    # Robust caps (winsorize at p99 to reduce extreme tails)
    p99 = out["play_time_hours"].quantile(0.99)
    out["play_time_hours_capped"] = np.minimum(out["play_time_hours"], p99)

    # Bucketize playtime for simple categorical features (quartiles)
    out["play_bucket_q"] = pd.qcut(
        out["play_time_hours"].rank(method="first"),  # avoid duplicate edges
        q=4, labels=[0, 1, 2, 3]
    ).astype("int8")

    # Simple label proxy (for ranking tasks later)
    out["label_played"] = (out["play_time_hours"] > 1.0).astype("int8")
    out["label_purchased_or_played"] = ((out["label_played"] == 1) | (out["purchased_flag"] == 1)).astype("int8")
    return out

def save_features(df: pd.DataFrame) -> None:
    dir_features().mkdir(parents=True, exist_ok=True)
    cols = [
        "user_id","game_id","play_time_hours","purchased_flag",
        "play_time_hours_log1p","play_time_hours_sqrt","play_time_hours_capped","play_bucket_q",
        "label_played","label_purchased_or_played","first_ts","last_ts"
    ]
    df[cols].to_parquet(path_interactions_features(), index=False)

def run() -> None:
    df = load_interactions()
    feats = add_transforms(df)
    save_features(feats)
    print(f"[features] OK → {path_interactions_features()}")

if __name__ == "__main__":
    run()
