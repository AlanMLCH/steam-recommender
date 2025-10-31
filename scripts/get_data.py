import requests
import json
from pathlib import Path
import time

STEAMSPY_API_URL = "http://steamspy.com/api.php"

def get_all_games():
    """Fetches all games from the SteamSpy API."""
    games = []
    page = 0
    while True:
        print(f"Fetching page {page}...")
        response = requests.get(STEAMSPY_API_URL, params={"request": "all", "page": page})
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            break
        data = response.json()
        if not data:
            break
        games.extend(data.values())
        page += 1
        time.sleep(1)  # Respect the API rate limit
    return games

def main():
    """Main function to get and save the data."""
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    games_file = raw_data_path / "games.json"

    if games_file.exists():
        print("Raw games data already exists. Skipping download.")
    else:
        print("Fetching games data from SteamSpy...")
        games = get_all_games()
        with open(games_file, "w") as f:
            json.dump(games, f, indent=2)
        print(f"Saved {len(games)} games to {games_file}")

    # Placeholder for Steam API integration
    print("\n--- Steam API Integration Placeholder ---")
    print("To get user data, you would need a Steam API key.")
    print("You would typically use the ISteamUser/GetFriendList and IPlayerService/GetOwnedGames endpoints.")
    print("This requires user consent and is beyond the scope of this initial implementation.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
