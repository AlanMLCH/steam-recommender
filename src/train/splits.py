from pathlib import Path
import os
import numpy as np
import pandas as pd
from typing import Tuple
from ..data_pipeline.config import path_interactions_features

def _env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)

def load_data() -> pd.DataFrame:
    return pd.read_parquet(path_interactions_features())

def _last_item_per_user(df: pd.DataFrame) -> pd.DataFrame:
    # use max play_time_hours as proxy for "most engaged" item
    # (you can replace with timestamp once you have it)
    df = df.sort_values(["user_id", "play_time_hours"], ascending=[True, False])
    last = df.groupby("user_id", as_index=False).first()[["user_id", "game_id"]]
    last = last.rename(columns={"game_id": "val_item"})
    return last

def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_pairs = _last_item_per_user(df)
    df_merged = df.merge(val_pairs, on="user_id", how="left")
    train = df_merged[df_merged["game_id"] != df_merged["val_item"]].copy()
    val = df_merged.drop_duplicates(["user_id"])[["user_id", "val_item"]].copy()
    val = val.rename(columns={"val_item": "game_id"})
    return train, val

def save_mappings(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # item count & user count help build embedding tables
    df[["user_id"]].drop_duplicates().to_parquet(out_dir / "users_in_data.parquet", index=False)
    df[["game_id"]].drop_duplicates().to_parquet(out_dir / "items_in_data.parquet", index=False)

def run(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    train, val = train_val_split(data)
    save_mappings(data, out_dir)
    return train[["user_id", "game_id"]], val[["user_id", "game_id"]]
