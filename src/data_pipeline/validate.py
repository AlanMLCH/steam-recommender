import json
from typing import Dict
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series, DataFrame
from .config import path_interactions, path_validation_report

class InteractionsSchema(pa.DataFrameModel):
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
