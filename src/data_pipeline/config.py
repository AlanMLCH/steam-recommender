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
