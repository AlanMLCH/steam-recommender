.PHONY: build pipeline train up down

IMAGE_NAME := game-rec-stage1
TAG := latest

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

pipeline:
	docker run --rm \
		-v ./data:/app/data \
		-v ./src:/app/src \
		$(IMAGE_NAME):$(TAG) \
		python -m src.data_pipeline.pipeline --step all

train:
	docker run --rm \
		-v ./data:/app/data \
		-v ./artifacts:/app/artifacts \
		$(IMAGE_NAME):$(TAG) \
		python -m src.train.run_training

up:
	docker-compose -f docker-compose.dashboard.yml up -d

down:
	docker-compose -f docker-compose.dashboard.yml down
