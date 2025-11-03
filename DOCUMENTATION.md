# Documentation

This document provides a more detailed explanation of the Game Recommender System.

## Architecture

(Same as before)

## Directory Structure

(Same as before)

## Data Acquisition

To train the model, you first need to acquire the data. We have provided a data acquisition pipeline that fetches data from the SteamSpy and Steam APIs and preprocesses it.

### Prerequisites

Before running the pipeline, you need to create a `.env` file in the root of the project with your Steam API key and a starting user ID:

```
STEAM_API_KEY=YOUR_STEAM_API_KEY
STEAM_USER_ID=YOUR_STEAM_USER_ID
```

You can get a Steam API key from the [Steam Community developer page](https://steamcommunity.com/dev/apikey).

### Running the Pipeline

To run the data acquisition pipeline, use the following command:

```bash
make data
```

This will run the `pipeline` service defined in `docker-compose.yml`. This service will execute the following scripts:

1.  `scripts/get_data.py`: Fetches a list of all games from the SteamSpy API and crawls the Steam API for user-item interactions.
2.  `scripts/preprocess_data.py`: Processes the raw data and creates `data/processed/items.csv` and `data/processed/interactions.csv`.

## Usage

(Same as before)
