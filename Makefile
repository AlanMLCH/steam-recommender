.PHONY: build up down data train index api test

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

data:
	python scripts/get_data.py
	python scripts/preprocess_data.py

train:
	docker run --rm -v $(PWD):/app -w /app --entrypoint python \
		$(shell docker build -q -f docker/Dockerfile.train .) src/modeling/train.py

index:
	docker run --rm -v $(PWD):/app -w /app --entrypoint python \
		$(shell docker build -q -f docker/Dockerfile.train .) src/index/build_faiss.py

api:
	docker compose up api

test:
	pytest -q