import requests
import json
from pathlib import Path
import time
import os

STEAMSPY_API_URL = "http://steamspy.com/api.php"
STEAM_API_URL = "http://api.steampowered.com"

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

def get_owned_games(user_id, api_key):
    """Fetches the owned games of a user."""
    response = requests.get(f"{STEAM_API_URL}/IPlayerService/GetOwnedGames/v1/", params={
        "key": api_key,
        "steamid": user_id,
        "include_appinfo": True,
        "format": "json"
    })
    if response.status_code != 200:
        return None
    return response.json().get("response", {}).get("games", [])

def get_friend_list(user_id, api_key):
    """Fetches the friend list of a user."""
    response = requests.get(f"{STEAM_API_URL}/ISteamUser/GetFriendList/v1/", params={
        "key": api_key,
        "steamid": user_id,
        "relationship": "friend"
    })
    if response.status_code != 200:
        return None
    return response.json().get("friendslist", {}).get("friends", [])

def steam_crawler(start_user_id, api_key, max_users=10):
    """Crawls Steam users to get their owned games."""
    users_to_visit = [start_user_id]
    visited_users = set()
    interactions = []

    while users_to_visit and len(visited_users) < max_users:
        current_user = users_to_visit.pop(0)
        if current_user in visited_users:
            continue

        print(f"Crawling user: {current_user}")
        owned_games = get_owned_games(current_user, api_key)
        if owned_games:
            for game in owned_games:
                interactions.append({
                    "user_id": current_user,
                    "item_id": game["appid"],
                    "playtime_forever": game["playtime_forever"]
                })
        
        visited_users.add(current_user)

        friend_list = get_friend_list(current_user, api_key)
        if friend_list:
            for friend in friend_list:
                if friend["steamid"] not in visited_users:
                    users_to_visit.append(friend["steamid"])
        
        time.sleep(1) # Rate limiting

    return interactions

def main():
    """Main function to get and save the data."""
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    games_file = raw_data_path / "games.json"
    interactions_file = raw_data_path / "interactions.json"

    # SteamSpy data
    if games_file.exists():
        print("Raw games data already exists. Skipping download.")
    else:
        print("Fetching games data from SteamSpy...")
        games = get_all_games()
        with open(games_file, "w") as f:
            json.dump(games, f, indent=2)
        print(f"Saved {len(games)} games to {games_file}")

    # Steam data
    api_key = os.environ.get("STEAM_API_KEY")
    start_user_id = os.environ.get("STEAM_USER_ID")

    if not api_key or not start_user_id:
        print("\nSteam API key or starting user ID not found in environment variables.")
        print("Skipping Steam data acquisition.")
        print("Please set STEAM_API_KEY and STEAM_USER_ID to fetch real interaction data.")
    elif interactions_file.exists():
        print("Raw interactions data already exists. Skipping download.")
    else:
        print("\nFetching interactions data from Steam...")
        interactions = steam_crawler(start_user_id, api_key)
        with open(interactions_file, "w") as f:
            json.dump(interactions, f, indent=2)
        print(f"Saved {len(interactions)} interactions to {interactions_file}")

if __name__ == "__main__":
    main()