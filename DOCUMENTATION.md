# Documentation

This document provides a more detailed explanation of the Game Recommender System.

## Architecture

The system is composed of the following main components:

-   **Two-Tower Model**: A deep learning model that learns separate representations (embeddings) for users and items (games). The model is trained to produce similar embeddings for users and the items they have interacted with.
-   **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors. We use it to build an index of the item embeddings, allowing for fast retrieval of the most similar items to a given user embedding.
-   **FastAPI Backend**: A Python web framework for building APIs. The backend exposes an endpoint for getting recommendations for a given user.
-   **Streamlit Dashboard**: A simple web application for interacting with the recommender system's API.

## Directory Structure

-   `configs/`
    -   `training.yaml`: Configuration for the model training process (e.g., learning rate, batch size, epochs).
    -   `service.yaml`: Configuration for the backend service (e.g., paths to the model and index).
-   `data/`
    -   `raw/`: For storing raw data.
    -   `processed/`: For storing processed data ready for training.
-   `artifacts/`
    -   `index.faiss`: The trained FAISS index.
    -   `item_emb.npy`: The item embeddings.
    -   `item_meta.parquet`: Metadata about the items.
-   `models/`
    -   `model.pt`: The trained PyTorch model checkpoint.
-   `notebooks/`
    -   `01_eda.ipynb`: Notebook for exploratory data analysis.
    -   `02_train.ipynb`: Notebook for training the model and generating artifacts.
    -   `03_offline_eval.ipynb`: Notebook for offline evaluation of the model.
-   `src/`
    -   `modeling/`: Source code for the two-tower model, losses, and training loop.
    -   `index/`: Source code for building and loading the FAISS index.
    -   `service/`: Source code for the FastAPI application and inference logic.
-   `dashboard/`
    -   `app.py`: The Streamlit dashboard application.
-   `tests/`
    -   `test_index.py`: Tests for the FAISS index.
    -   `test_inference.py`: Tests for the API.
-   `docker/`
    -   `Dockerfile.api`: Dockerfile for the FastAPI backend.
    -   `Dockerfile.train`: Dockerfile for the training environment.
    -   `Dockerfile.dashboard`: Dockerfile for the Streamlit dashboard.
-   `docker-compose.yml`: Docker Compose file for running the services.
-   `Makefile`: Makefile with convenience commands for building, running, and training.

## Data Acquisition

To train the model, you first need to acquire the data. We have provided scripts to fetch data from the SteamSpy API and preprocess it.

To run the data acquisition process, use the following command:

```bash
make data
```

This will execute the following scripts:

1.  `scripts/get_data.py`: Fetches a list of all games from the SteamSpy API and saves it to `data/raw/games.json`.
2.  `scripts/preprocess_data.py`: Processes the raw game data and creates `data/processed/items.csv` and a dummy `data/processed/interactions.csv`.

**Note on API Keys:** The SteamSpy API is public, but the Steam API requires an API key for fetching user data. The `get_data.py` script includes a placeholder for Steam API integration, but it is not fully implemented. To get real user data, you would need to obtain a Steam API key and implement the logic to fetch user profiles and their owned games.

## Usage

### Getting Recommendations

You can get recommendations by sending a POST request to the `/recommend` endpoint. For example, using `curl`:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "time_played": {"g_221": 120.0},
    "genres": ["rpg", "roguelike"],
    "top_k": 5
  }'
```

This will return a JSON response with the recommended items for the user.

### Training

To train the model, you need to have your data in the `data/processed` directory. The training process is defined in the `notebooks/02_train.ipynb` notebook. You can run the training in a Docker container using the `make train` command.

This will execute the `src/modeling/train.py` script, which will:

1.  Load the data.
2.  Train the two-tower model.
3.  Save the trained model to `models/model.pt`.
4.  Generate item embeddings and save them to `artifacts/item_emb.npy`.

### Indexing

After training the model and generating the item embeddings, you need to build the FAISS index. You can do this by running:

```bash
make index
```

This will execute the `src/index/build_faiss.py` script, which will:

1.  Load the item embeddings.
2.  Build the FAISS index.
3.  Save the index to `artifacts/index.faiss`.