"""
FastAPI application that serves:
    - /health      : quick check that the API is alive
    - /live        : liveness probe
    - /ready       : readiness probe
    - /info        : model + dataset metadata
    - /search      : semantic search over embedded abstracts
    - /metrics     : simple JSON metrics
    - /metrics-prom: Prometheus-style metrics

This is the online serving layer of the MLOps pipeline.
"""

import json
import logging
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    MODEL_NAME,
    EMBEDDINGS_FILE,
    IDS_FILE,
    METADATA_FILE,
)
from pipeline.data_loader import load_abstracts


# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("semantic-search")


# --------------------------------------------------
# Load embeddings, metadata, and model at import time
# --------------------------------------------------
logger.info("Loading embeddings and metadata...")
embeddings = np.load(EMBEDDINGS_FILE)
ids = np.load(IDS_FILE)
with open(METADATA_FILE) as f:
    metadata = json.load(f)

# Pre-compute mean embedding for simple drift detection
train_mean_embedding = np.mean(embeddings, axis=0)
logger.info("Computed mean training embedding for drift detection")

logger.info("Loading model for query encoding...")
model = SentenceTransformer(MODEL_NAME)

start_time = time.time()


# --------------------------------------------------
# Simple in-memory metrics
# --------------------------------------------------
metrics = {
    "request_count": {
        "/health": 0,
        "/info": 0,
        "/search": 0,
        "/metrics": 0,
        "/metrics-prom": 0,
        "/live": 0,
        "/ready": 0,
    },
    "error_count": {
        "/search": 0,
    },
    "latency": {
        "/search": [],
    },
    "drift_count": 0,
}


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def record_request(endpoint: str) -> None:
    if endpoint in metrics["request_count"]:
        metrics["request_count"][endpoint] += 1


def record_error(endpoint: str) -> None:
    if endpoint in metrics["error_count"]:
        metrics["error_count"][endpoint] += 1


def record_latency(endpoint: str, value_ms: float) -> None:
    if endpoint in metrics["latency"]:
        metrics["latency"][endpoint].append(value_ms)


def detect_drift(query_vec: np.ndarray, mean_vec: np.ndarray, threshold: float = 0.20) -> bool:
    """
    Simple drift detection:
    - Compute cosine similarity between query embedding and
      mean training embedding.
    - If similarity is lower than threshold → treat as drift.
    """
    q = query_vec / np.linalg.norm(query_vec)
    m = mean_vec / np.linalg.norm(mean_vec)

    similarity = float(np.dot(q, m))
    logger.info(f"drift_cosine={similarity:.3f}")

    return similarity < threshold


def generate_prometheus_metrics() -> str:
    """
    Convert internal metrics dict into Prometheus text exposition format.
    """
    lines: list[str] = []

    # Request counters
    lines.append("# HELP ml_requests_total Total number of requests per endpoint")
    lines.append("# TYPE ml_requests_total counter")
    for endpoint, count in metrics["request_count"].items():
        lines.append(f'ml_requests_total{{endpoint="{endpoint}"}} {count}')

    # Error counters
    lines.append("# HELP ml_errors_total Total number of errors per endpoint")
    lines.append("# TYPE ml_errors_total counter")
    for endpoint, count in metrics["error_count"].items():
        lines.append(f'ml_errors_total{{endpoint="{endpoint}"}} {count}')

    # Drift counter
    lines.append("# HELP ml_drift_total Number of drifted input queries")
    lines.append("# TYPE ml_drift_total counter")
    lines.append(f"ml_drift_total {metrics['drift_count']}")

    # Latency histogram
    lines.append("# HELP ml_latency_ms Latency of search requests in milliseconds")
    lines.append("# TYPE ml_latency_ms histogram")

    buckets = [10, 50, 100, 250, 500]  # ms
    latency_values = metrics["latency"]["/search"]

    for b in buckets:
        count_le = sum(1 for v in latency_values if v <= b)
        lines.append(f'ml_latency_ms_bucket{{endpoint="/search",le="{b}"}} {count_le}')

    lines.append(
        f'ml_latency_ms_bucket{{endpoint="/search",le="+Inf"}} {len(latency_values)}'
    )

    total_latency = sum(latency_values)
    lines.append(f'ml_latency_ms_sum{{endpoint="/search"}} {total_latency}')
    lines.append(f'ml_latency_ms_count{{endpoint="/search"}} {len(latency_values)}')

    return "\n".join(lines) + "\n"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/health")
def health():
    record_request("/health")
    logger.info("/health called")
    return {"status": "ok"}


@app.get("/live")
def live():
    record_request("/live")
    logger.info("/live called")
    return {"alive": True}


@app.get("/ready")
def ready():
    record_request("/ready")
    model_loaded = model is not None
    embeddings_loaded = embeddings is not None and ids is not None
    is_ready = model_loaded and embeddings_loaded
    logger.info(f"/ready | ready={is_ready}")
    return {"ready": is_ready}


@app.get("/info")
def info():
    record_request("/info")
    uptime = time.time() - start_time
    logger.info("/info called")
    return {
        "model_name": metadata.get("model_name", MODEL_NAME),
        "num_documents": metadata.get("num_documents", len(ids)),
        "embedding_dimension": metadata.get(
            "embedding_dimension", int(embeddings.shape[1])
        ),
        "uptime_seconds": int(uptime),
        "started_at": datetime.fromtimestamp(start_time).isoformat(),
    }


@app.post("/search")
def search(request: SearchRequest):
    record_request("/search")
    t0 = time.time()

    logger.info(f'/search | query="{request.query}" | top_k={request.top_k}')

    try:
        query_vec = model.encode([request.query])[0]

        # Drift detection
        if detect_drift(query_vec, train_mean_embedding):
            metrics["drift_count"] += 1
            logger.warning(f"DRIFT DETECTED | query='{request.query}'")

        # Cosine similarity search
        norm_query = query_vec / np.linalg.norm(query_vec)
        norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        scores = norm_emb @ norm_query
        top_idx = np.argsort(scores)[::-1][: request.top_k]

        results = [
            {"id": int(ids[i]), "score": float(scores[i])}
            for i in top_idx
        ]

    except Exception as e:
        record_error("/search")
        logger.error(f'/search ERROR | query="{request.query}" | exception="{e}"')
        return {"error": str(e)}

    latency_ms = (time.time() - t0) * 1000.0
    record_latency("/search", latency_ms)

    logger.info(
        f'/search completed | query="{request.query}" '
        f"| results={len(results)} | latency={latency_ms:.2f}ms"
    )

    return {
        "query": request.query,
        "results": results,
        "latency_ms": latency_ms,
    }


@app.get("/metrics")
def get_metrics():
    record_request("/metrics")
    logger.info("/metrics called")
    return metrics


@app.get("/metrics-prom", response_class=PlainTextResponse)
def metrics_prom():
    record_request("/metrics-prom")
    logger.info("/metrics-prom called")
    return generate_prometheus_metrics()
