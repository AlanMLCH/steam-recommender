import streamlit as st, requests, json

st.title("Game Recommender — Operator Console")
api = st.text_input("API base URL", "http://api:8000")
user_id = st.text_input("User ID", "u_123")
genres = st.multiselect("Genres", ["rpg","fps","roguelike","racing","strategy","simulation","sports","adventure","puzzle"], default=["rpg","roguelike"])
top_k = st.slider("Top-K", 1, 20, 10)

if st.button("Get Recommendations"):
    payload = {"user_id": user_id, "time_played": {"g_221": 120.0}, "genres": genres, "achievements": {}, "top_k": top_k}
    r = requests.post(f"{api}/recommend", json=payload, timeout=30)
    st.json(r.json())