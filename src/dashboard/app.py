import streamlit as st
import requests
import pandas as pd

# API endpoint
API_BASE_URL = "http://api:8000"

def get_recommendations(game_ids, top_k=10, exclude_seen=True):
    """Calls the recommendation API."""
    payload = {
        "game_ids": game_ids,
        "top_k": top_k,
        "exclude_seen": exclude_seen
    }
    try:
        response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        response.raise_for_status()
        return response.json()["items"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling recommend API: {e}")
        return []

def search_games(query: str, limit: int = 10):
    """Calls the game search API."""
    if not query:
        return []
    try:
        response = requests.get(f"{API_BASE_URL}/games/search", params={"q": query, "limit": limit})
        response.raise_for_status()
        return response.json()["items"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling search API: {e}")
        return []

def main():
    st.title("Steam Game Recommender")

    # --- Initialize session state for selected games ---
    if 'selected_games' not in st.session_state:
        st.session_state.selected_games = {}  # Store as dict {title: game_id}

    # --- Game Selection ---
    st.header("Select Games You Like")

    # --- Search for games ---
    search_query = st.text_input("Search for games to add to your list:", "")
    if search_query:
        search_results = search_games(search_query)
        if search_results:
            st.write("Search Results:")
            for game in search_results:
                title = game["title"]
                game_id = game["game_id"]
                if st.button(f"Add '{title}'"):
                    st.session_state.selected_games[title] = game_id
                    st.rerun() # Rerun to update the display of selected games

    # --- Display selected games ---
    st.subheader("Your selected games:")
    if not st.session_state.selected_games:
        st.info("No games selected yet. Use the search bar above to find and add games.")
    else:
        for title, game_id in list(st.session_state.selected_games.items()):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- {title} (ID: {game_id})")
            with col2:
                if st.button(f"Remove##{game_id}", key=f"remove_{game_id}"):
                    del st.session_state.selected_games[title]
                    st.rerun()

    selected_ids = list(st.session_state.selected_games.values())

    # --- Recommendation Parameters ---
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K Recommendations", min_value=5, max_value=50, value=10)
    exclude_seen = st.sidebar.checkbox("Exclude already selected games?", value=True)

    # --- Generate Recommendations ---
    if st.button("Get Recommendations"):
        if not selected_ids:
            st.warning("Please select at least one game.")
        else:
            with st.spinner("Fetching recommendations..."):
                recs = get_recommendations(selected_ids, top_k, exclude_seen)

                if recs:
                    st.header("Recommended for you:")
                    df = pd.DataFrame(recs)
                    for _, row in df.iterrows():
                        st.subheader(row["title"])
                        st.write(f"**Game ID:** {row['game_id']} | **Score:** {row['score']:.4f}")
                    st.success("Done!")
                else:
                    st.error("Could not retrieve recommendations.")


if __name__ == "__main__":
    main()
