# Steam Recommender (Two‑Tower + FAISS + Reranker)


## 1) Train & build index (first time)
make train


## 2) Start the stack
make up


- Gateway API: http://localhost:8000/recommend?user_id=123&k=50&topn=10
- Traefik dashboard: http://localhost:8080


## 3) Canary & shadow
# 10% traffic to challenger
make canary W=10
# rollback to 0% challenger
make weights
# switch blue/green explicitly
make blue # 100% champion
make green # 100% challenger (use with caution)


## 4) Nightly index build & drift
- Cron container rebuilds FAISS nightly at 02:00.
- Hourly PSI check writes `artifacts/retrain_requested` if drift > 0.25.
- Hook your CI/CD to detect that file and run `make train` to retrain and redeploy.


## 5) Metrics & auto‑rollback
- Gateway exposes `/metrics_json` with p95 latency and a toy KPI.
- Watchdog watches p95 vs `SLA_P95_MS` (default 150 ms) and KPI; auto‑rolls back to 100% champion after 3 bad intervals.


## 6) Data
- Put your Steam user‑game interactions under `./data/`. Adapt `models/two_tower_train.py` to load real data.


## 7) Notes
- TorchServe handlers are minimal; replace with real rerankers (e.g., cross‑encoder or feature MLP).
- This template avoids heavy infra (Prometheus/Kafka/Airflow). You can plug them in later.
- Windows users: run from WSL2 for best Docker experience.