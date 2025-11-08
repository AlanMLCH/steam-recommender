Normalize embeddings → cosine similarity

emb = ITEM_EMB / np.linalg.norm(ITEM_EMB, axis=1, keepdims=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype(np.float32))


This often boosts Recall@K by 5-10 points.

Weighted user embedding

weights = np.linspace(1.0, 2.0, num=len(game_ids))  # emphasize recent
u_vec = (ITEM_EMB[game_ids] * weights[:, None]).mean(0)


Expose metadata in responses

items_df = pd.read_parquet("/app/data/lookups/item_lookup.parquet")
joined = items_df.merge(pd.DataFrame(results, columns=["game_id","score"]))


Return {title, score} instead of IDs.

Streamlit or simple HTML dashboard

Form to input liked games (game_id or title)

Calls /recommend

Displays thumbnails & titles of top-N results

Offline reranker (optional later)

Tiny cosine-MLP or cross-encoder to rerank FAISS top-50.