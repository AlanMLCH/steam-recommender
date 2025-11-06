Stage A, Step 1: ingest + clean + validate + export to Parquet from the Kaggle Steam-200k dataset. Below is a minimal, Windows-friendly starter you can drop into a repo and run.

Parses the raw CSV (Steam-200k).

Normalizes game titles (trim/space collapse).

Aggregates to one row per (user, game) with:

play_time_hours (sum of “play” hours),

purchased_flag (1 if any “purchase” seen),

optional first_ts/last_ts left empty for now (placeholder columns for later).

Creates integer ids: user_id and game_id (stable via lookups).

Validates with pandera (basic schema).

Saves:

data/processed/interactions.parquet

data/lookups/user_lookup.parquet (original_user → user_id)

data/lookups/item_lookup.parquet (game_title → game_id)

data/reports/ingest_report.json (counts, density, nulls, basic stats)

LAYOUT

game-rec-stage1/
  data/
    raw/steam-200k.csv      # place Kaggle file here
    processed/              # created by pipeline
    reports/                # created by pipeline
    lookups/                # created by pipeline
    features/               # created by pipeline
  src/
    __init__.py
    config.py
    extract.py
    validate.py
    feature_engineering.py
    pipeline.py
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env                      

--------------------------------------------------
CODE:

dockerfile:

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but handy for pandas/pyarrow perf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/

# Default command runs the whole pipeline (override to run step-by-step)
CMD ["python", "src/pipeline.py", "--step", "all"]


docker-compose.yml:

version: "3.9"
services:
  pipeline:
    build: .
    image: game-rec-stage1:latest
    environment:
      DATA_DIR: /app/data
    volumes:
      - ./data:/app/data
    command: ["python", "src/pipeline.py", "--step", "all"]


src/config.py:

import os
from pathlib import Path

def data_dir() -> Path:
    root = os.environ.get("DATA_DIR", "/app/data")
    return Path(root)

def path_raw_csv() -> Path:
    return data_dir() / "raw" / "steam-200k.csv"

def dir_processed() -> Path:
    return data_dir() / "processed"

def dir_lookups() -> Path:
    return data_dir() / "lookups"

def dir_reports() -> Path:
    return data_dir() / "reports"

def dir_features() -> Path:
    return data_dir() / "features"

def path_interactions() -> Path:
    return dir_processed() / "interactions.parquet"

def path_user_lookup() -> Path:
    return dir_lookups() / "user_lookup.parquet"

def path_item_lookup() -> Path:
    return dir_lookups() / "item_lookup.parquet"

def path_ingest_report() -> Path:
    return dir_reports() / "ingest_report.json"

def path_validation_report() -> Path:
    return dir_reports() / "validation_report.json"

def path_interactions_features() -> Path:
    return dir_features() / "interactions_features.parquet"


src/validate.py:

import json
from typing import Dict
import pandas as pd
import pandera as pa
from pandera.typing import Series, DataFrame
from .config import path_interactions, path_validation_report

class InteractionsSchema(pa.SchemaModel):
    user_id: Series[int] = pa.Field(ge=0)
    game_id: Series[int] = pa.Field(ge=0)
    play_time_hours: Series[float] = pa.Field(ge=0)
    purchased_flag: Series[int] = pa.Field(isin=[0, 1])
    first_ts: Series[pd.Timestamp] = pa.Field(nullable=True)
    last_ts: Series[pd.Timestamp] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = True

def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(path_interactions())

def validate_interactions(df: pd.DataFrame) -> DataFrame[InteractionsSchema]:
    return InteractionsSchema.validate(df, lazy=True)

def make_report(df: pd.DataFrame) -> Dict:
    return {
        "rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_items": int(df["game_id"].nunique()),
        "matrix_density": float(len(df) / max(1, df["user_id"].nunique() * df["game_id"].nunique())),
        "play_time_hours": {
            "min": float(df["play_time_hours"].min()),
            "p50": float(df["play_time_hours"].quantile(0.5)),
            "mean": float(df["play_time_hours"].mean()),
            "max": float(df["play_time_hours"].max()),
        },
        "purchased_rate": float(df["purchased_flag"].mean()),
        "null_counts": df.isna().sum().to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }

def save_report(report: Dict) -> None:
    path_validation_report().parent.mkdir(parents=True, exist_ok=True)
    with open(path_validation_report(), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

def run() -> None:
    df = load_interactions()
    _ = validate_interactions(df)
    report = make_report(df)
    save_report(report)
    print(f"[validate] OK → {path_validation_report()}")

if __name__ == "__main__":
    run()


src/extract.py
import json
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
from .config import (
    path_raw_csv, dir_processed, dir_lookups, path_interactions,
    path_user_lookup, path_item_lookup, path_ingest_report
)

def read_steam_200k(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        cols = [c.lower().strip() for c in df.columns]
        df.columns = cols
        rename_map = {}
        if "user_id" in cols: rename_map["user_id"] = "user"
        if "username" in cols and "user" not in cols: rename_map["username"] = "user"
        if "game" in cols: rename_map["game"] = "game_title"
        if "title" in cols and "game_title" not in cols: rename_map["title"] = "game_title"
        if "action" in cols: rename_map["action"] = "behavior"
        if "behaviour" in cols and "behavior" not in cols: rename_map["behaviour"] = "behavior"
        if "playtime_forever" in cols and "hours" not in cols: rename_map["playtime_forever"] = "hours"
        if rename_map: df = df.rename(columns=rename_map)
    except Exception:
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
    csv = path_raw_csv()
    if not csv.exists():
        raise FileNotFoundError(f"Missing raw CSV at {csv}")
    raw = read_steam_200k(csv)
    raw = normalize_titles(raw)
    agg = aggregate_interactions(raw)
    interactions, users, items = build_id_maps(agg)
    save_parquets(interactions, users, items)
    save_ingest_report(build_ingest_report(interactions))
    print(f"[extract] OK → {path_interactions()}")

if __name__ == "__main__":
    run()


src/feature_engineering.py
import pandas as pd
import numpy as np
from .config import path_interactions, path_interactions_features, dir_features

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

🧩 src/pipeline.py
import argparse
from . import extract, validate, feature_engineering

def parse_args():
    p = argparse.ArgumentParser(description="Stage A pipeline runner")
    p.add_argument("--step", choices=["extract", "validate", "features", "all"], default="all")
    return p.parse_args()

def run_step(step: str) -> None:
    if step == "extract":
        extract.run()
    elif step == "validate":
        validate.run()
    elif step == "features":
        feature_engineering.run()
    elif step == "all":
        extract.run()
        validate.run()
        feature_engineering.run()

def main():
    args = parse_args()
    run_step(args.step)

if __name__ == "__main__":
    main()


requirements.txt

pandas>=2.1.0
pyarrow>=14.0.0
pandera>=0.20.0
