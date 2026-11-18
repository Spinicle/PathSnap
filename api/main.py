"""FastAPI backend, runs on Laptop 1 (Owner: D).
Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Ground truth stays here; responses carry only predictions + computed metrics.
"""
import logging
import time

from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s from %s -> %s (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code,
             (time.perf_counter() - t0) * 1000)
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    # TODO: list_trips("eval") once PATHSNAP_EVAL_DATA is set
    return {"trips": []}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    # TODO: load trip (no route) -> candidates -> features -> model -> viterbi
    #       then score against server-side route. Response shape: docs/CONTRACTS.md
    raise HTTPException(501, "pipeline not wired yet")
