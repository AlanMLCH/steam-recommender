import streamlit as st
import requests
import pandas as pd

# API endpoint
API_URL = "http://api:8000/recommend"

def get_recommendations(game_ids, top_k=10, exclude_seen=True):
    """Calls the recommendation API."""
    payload = {
        "game_ids": game_ids,
        "top_k": top_k,
        "exclude_seen": exclude_seen
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["items"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")
        return []

def main():
    st.title("Steam Game Recommender")

    # --- Game Selection ---
    st.header("Select Games You Like")
    

    mock_games = {
        "Half-Life 2": 220,
        "Portal 2": 620,
        "Left 4 Dead 2": 550,
        "Counter-Strike: Global Offensive": 730,
        "The Elder Scrolls V: Skyrim": 72850,
        "Terraria": 105600,
        "Stardew Valley": 413150,
        "Sid Meier's Civilization V": 8930,
    }
    
    selected_titles = st.multiselect(
        "Choose one or more games:",
        options=list(mock_games.keys())
    )
    
    selected_ids = [mock_games[title] for title in selected_titles]

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
                    
                    # Display results in a more appealing way
                    df = pd.DataFrame(recs)
                    
                    for _, row in df.iterrows():
                        st.subheader(row["title"])
                        st.write(f"**Game ID:** {row['game_id']} | **Score:** {row['score']:.4f}")
                        # You could add game thumbnails here if you have image URLs
                        # st.image(get_game_image_url(row['game_id']))
                    
                    st.success("Done!")
                else:
                    st.error("Could not retrieve recommendations.")

if __name__ == "__main__":
    main()
