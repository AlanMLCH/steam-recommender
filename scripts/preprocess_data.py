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

def create_dummy_interactions(processed_data_path):
    """Creates a dummy interactions file."""
    interactions_file = processed_data_path / "interactions.csv"

    if interactions_file.exists():
        print("Dummy interactions data already exists. Skipping creation.")
        return

    print("Creating dummy interactions data...")
    # Create a small dummy dataset
    interactions = {
        "user_id": ["u_1", "u_1", "u_2", "u_2", "u_3"],
        "item_id": [220, 570, 730, 570, 220],
    }
    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved dummy interactions to {interactions_file}")
    print("NOTE: This is a dummy dataset. You should replace it with real user-item interaction data.")

def main():
    """Main function to preprocess the data."""
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    preprocess_games(raw_data_path, processed_data_path)
    create_dummy_interactions(processed_data_path)

if __name__ == "__main__":
    main()
