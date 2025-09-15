import os, time, json, statistics
from typing import List, Dict
from fastapi import FastAPI, Query
import httpx
from prometheus_client import Summary, Counter, Gauge


RERANK_URL = os.getenv("RERANK_URL", "http://traefik/rerank")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8000")
SLA_P95_MS = int(os.getenv("SLA_P95_MS", "150"))


app = FastAPI(title="Gateway API")


REQ_LAT = [] # rolling window of latencies (ms)
SUCCESS = Counter('req_success_total', 'Successful rec requests')
FAIL = Counter('req_fail_total', 'Failed rec requests')
P95_G = Gauge('latency_p95_ms', 'p95 latency ms')
KPI_G = Gauge('business_kpi', 'Business KPI (conversion proxy)')


async def get_candidates(user_id: int, k: int) -> List[int]:
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"{RETRIEVAL_URL}/retrieve", params={"user_id": user_id, "k": k})
        r.raise_for_status()
        return r.json()["candidates"]


async def rerank(user_id: int, candidates: List[int]) -> List[int]:
    async with httpx.AsyncClient(timeout=5) as client:
        payload = {"user_id": user_id, "candidates": candidates}
        r = await client.post(f"{RERANK_URL}/predictions/rerank", json=payload)
        r.raise_for_status()
        return r.json()["ranked_ids"]


@app.get("/recommend")
async def recommend(user_id: int = Query(...), k: int = Query(50), topn: int = Query(10)):
    t0 = time.time()
    try:
        cands = await get_candidates(user_id, k)
        ranked = await rerank(user_id, cands)
        result = ranked[:topn]
        SUCCESS.inc()
        # simulate KPI: higher if user_id even
        kpi = 1.0 if (user_id % 2 == 0) else 0.8
        KPI_G.set(kpi)
        return {"user_id": user_id, "items": result}
    except Exception as e:
        FAIL.inc()
        return {"error": str(e)}
    finally:
        dt = (time.time() - t0) * 1000
        REQ_LAT.append(dt)
        if len(REQ_LAT) > 500:
            REQ_LAT.pop(0)
        if REQ_LAT:
            p95 = statistics.quantiles(REQ_LAT, n=20)[18]
            P95_G.set(p95)


@app.get("/metrics_json")
async def metrics_json():
    p95 = float(P95_G._value.get()) if P95_G._value.get() is not None else 0.0
    kpi = float(KPI_G._value.get()) if KPI_G._value.get() is not None else 0.0
    return {"p95_ms": p95, "kpi": kpi, "sla_ms": SLA_P95_MS}