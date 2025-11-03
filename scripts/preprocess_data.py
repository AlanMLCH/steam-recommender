import pandas as pd
import json
from pathlib import Path

def preprocess_games(raw_data_path, processed_data_path):
    """Processes the raw games data."""
    games_file = raw_data_path / "games.json"
    items_file = processed_data_path / "items.csv"

    if items_file.exists():
        print("Processed items data already exists. Skipping preprocessing.")
        return

    if not games_file.exists():
        print("Raw games data not found. Please run `scripts/get_data.py` first.")
        return

    with open(games_file, "r") as f:
        games_data = json.load(f)

    items = []
    for game in games_data:
        items.append({
            "item_id": game["appid"],
            "name": game["name"],
            "developer": game["developer"],
            "publisher": game["publisher"],
            "owners": game["owners"],
            "price": game["price"],
            "genres": game.get("genre", ""),
            "tags": ",".join(game.get("tags", {}).keys()),
        })

    items_df = pd.DataFrame(items)
    items_df.to_csv(items_file, index=False)
    print(f"Saved {len(items_df)} items to {items_file}")

def preprocess_interactions(raw_data_path, processed_data_path):
    """Processes the raw interactions data."""
    interactions_file = raw_data_path / "interactions.json"
    processed_interactions_file = processed_data_path / "interactions.csv"

    if processed_interactions_file.exists():
        print("Processed interactions data already exists. Skipping preprocessing.")
        return

    if not interactions_file.exists():
        print("Raw interactions data not found. Please run `scripts/get_data.py` first.")
        return

    with open(interactions_file, "r") as f:
        interactions_data = json.load(f)

    interactions_df = pd.DataFrame(interactions_data)
    interactions_df.to_csv(processed_interactions_file, index=False)
    print(f"Saved {len(interactions_df)} interactions to {processed_interactions_file}")

def main():
    """Main function to preprocess the data."""
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    preprocess_games(raw_data_path, processed_data_path)
    preprocess_interactions(raw_data_path, processed_data_path)

if __name__ == "__main__":
    main()