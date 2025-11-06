# Steam 200k Dataset ETL Pipeline

This project contains a simple ETL pipeline for the Steam 200k dataset.

## How to run the pipeline

### Prerequisites

-   Docker
-   Docker Compose

### 1. Get the data

1.  Download the Steam 200k dataset from Kaggle: [https://www.kaggle.com/tamber/steam-video-games](https://www.kaggle.com/tamber/steam-video-games)
2.  Place the `steam-200k.csv` file in the `data/raw` directory.

### 2. Build the Docker image

```bash
docker compose build
```

### 3. Run the pipeline

To run the entire ETL pipeline, use the following command:

```bash
docker compose run --rm pipeline
```

This will run all the steps of the pipeline: extract, validate, and features.

### 4. Output

The pipeline will create the following files and directories inside the `data` directory:

-   `data/processed/interactions.parquet`
-   `data/lookups/user_lookup.parquet`
-   `data/lookups/item_lookup.parquet`
-   `data/reports/ingest_report.json`
-   `data/reports/validation_report.json`
-   `data/features/interactions_features.parquet`
