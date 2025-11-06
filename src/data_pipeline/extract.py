import json
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
from .config import (
    path_raw_csv, dir_processed, dir_lookups, path_interactions,
    path_user_lookup, path_item_lookup, path_ingest_report
)

def read_steam_200k(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None,
                     names=["user", "game_title", "behavior", "hours", "value"])
    if "hours" not in df.columns: df["hours"] = 0.0
    if "value" not in df.columns: df["value"] = 0
    return df

def normalize_titles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_title"] = (
        out["game_title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return out

def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["behavior"] = d["behavior"].str.lower().str.strip()
    d["hours"] = pd.to_numeric(d["hours"], errors="coerce").fillna(0.0)

    play = (
        d.loc[d["behavior"] == "play", ["user", "game_title", "hours"]]
        .groupby(["user", "game_title"], as_index=False)["hours"].sum()
        .rename(columns={"hours": "play_time_hours"})
    )

    purch = (
        d.loc[d["behavior"] == "purchase", ["user", "game_title"]]
        .drop_duplicates()
        .assign(purchased_flag=1)
    )

    agg = play.merge(purch, how="left", on=["user", "game_title"])
    agg["purchased_flag"] = agg["purchased_flag"].fillna(0).astype(int)
    agg["first_ts"] = pd.NaT
    agg["last_ts"] = pd.NaT
    return agg

def build_id_maps(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = (
        agg[["user"]].drop_duplicates().sort_values("user", key=lambda s: s.astype(str))
        .reset_index(drop=True).reset_index(names="user_id")
    )
    items = (
        agg[["game_title"]].drop_duplicates()
        .sort_values("game_title", key=lambda s: s.astype(str))
        .reset_index(drop=True).reset_index(names="game_id")
    )
    interactions = (
        agg.merge(users, on="user").merge(items, on="game_title")
          .drop(columns=["user", "game_title"])
          .loc[:, ["user_id", "game_id", "play_time_hours", "purchased_flag", "first_ts", "last_ts"]]
          .astype({"user_id": "int32", "game_id": "int32"})
    )
    return interactions, users.rename(columns={"user": "original_user"}), items.rename(columns={"game_title": "title"})

def save_parquets(interactions: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None:
    dir_processed().mkdir(parents=True, exist_ok=True)
    dir_lookups().mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(path_interactions(), index=False)
    users.to_parquet(path_user_lookup(), index=False)
    items.to_parquet(path_item_lookup(), index=False)

def build_ingest_report(interactions: pd.DataFrame) -> Dict:
    n = len(interactions)
    return {
        "rows": int(n),
        "n_users": int(interactions["user_id"].nunique()),
        "n_items": int(interactions["game_id"].nunique()),
        "matrix_density": float(n / max(1, interactions["user_id"].nunique() * interactions["game_id"].nunique())),
        "purchased_rate": float(interactions["purchased_flag"].mean()),
    }

def save_ingest_report(report: Dict) -> None:
    path_ingest_report().parent.mkdir(parents=True, exist_ok=True)
    with open(path_ingest_report(), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

def run() -> None:
    csv_path = path_raw_csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing raw CSV at {csv_path}. Please download it from Kaggle and place it in the data/raw directory.")

    raw = read_steam_200k(csv_path)
    raw = normalize_titles(raw)
    agg = aggregate_interactions(raw)
    interactions, users, items = build_id_maps(agg)
    save_parquets(interactions, users, items)
    save_ingest_report(build_ingest_report(interactions))
    print(f"[extract] OK → {path_interactions()}")

if __name__ == "__main__":
    run()
