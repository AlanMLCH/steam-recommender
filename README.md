# Game Recommender System

This project is a game recommender system built with a two-tower model, FAISS for efficient similarity search, a FastAPI backend, and a Streamlit dashboard for interacting with the API.

## Project Structure

The project is organized into the following directories:

- `configs/`: Configuration files for training and service.
- `data/`: For storing raw and processed data (ignored by git).
- `artifacts/`: For storing the FAISS index and other generated files (ignored by git).
- `models/`: For storing trained model checkpoints (ignored by git).
- `notebooks/`: Jupyter notebooks for exploration, training, and evaluation.
- `src/`: Source code for the recommender system.
- `dashboard/`: The Streamlit dashboard application.
- `tests/`: Unit tests.
- `docker/`: Dockerfiles for the different services.

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- make (optional, for convenience)

### Running the Application

1.  **Build the Docker images:**

    ```bash
    make build
    ```

    or

    ```bash
    docker compose build
    ```

2.  **Start the services:**

    ```bash
    make up
    ```

    or

    ```bash
    docker compose up -d
    ```

    This will start the FastAPI backend and the Streamlit dashboard.

    -   FastAPI API is available at `http://localhost:8000`
    -   Streamlit dashboard is available at `http://localhost:8501`

3.  **Stopping the services:**

    ```bash
    make down
    ```

    or

    ```bash
    docker compose down
    ```

### Training the Model and Building the Index

(Note: You need to have data in the `data/processed` directory for this to work.)

1.  **Train the model:**

    ```bash
    make train
    ```

2.  **Build the FAISS index:**

    ```bash
    make index
    ```

### API

The API has a `/recommend` endpoint that accepts POST requests with the following format:

```json
{
  "user_id": "u_123",
  "time_played": {"g_221": 120.0, "g_501": 8.0},
  "genres": ["rpg", "roguelike", "pixel-art"],
  "achievements": {"g_221": 15, "g_777": 2},
  "top_k": 10
}
```
