import os
import time
import logging
import random

from fastapi import FastAPI, Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST


app = FastAPI(title="FraudGuard API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-api")

MODEL_VERSION = os.getenv("MODEL_VERSION", "stable")
logger.info(f"Starting FraudGuard API | version={MODEL_VERSION}")



http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

http_request_latency = Histogram(
    "http_request_latency_seconds",
    "End-to-end application latency (FastAPI)",
    ["method", "path"]
)

model_inference_latency = Histogram(
    "fraud_model_inference_latency_seconds",
    "Pure model inference latency",
    ["version"]
)

predictions_total = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["version"]
)

fraud_positive_total = Counter(
    "fraud_positive_predictions_total",
    "Fraud positive predictions",
    ["version"]
)

fraud_negative_total = Counter(
    "fraud_negative_predictions_total",
    "Fraud negative predictions",
    ["version"]
)


def predict_fraud(data: dict) -> int:
    time.sleep(0.01) 
    return 1 if data["amount"] > 1000 else 0

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    http_requests_total.labels(
        method=request.method,
        path=request.url.path,
        status=response.status_code
    ).inc()

    http_request_latency.labels(
        method=request.method,
        path=request.url.path
    ).observe(duration)

    return response

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(request: dict):
    amount = float(request["amount"])

    with model_inference_latency.labels(version=MODEL_VERSION).time():
        prediction = predict_fraud(request)

    predictions_total.labels(version=MODEL_VERSION).inc()

    if prediction == 1:
        fraud_positive_total.labels(version=MODEL_VERSION).inc()
    else:
        fraud_negative_total.labels(version=MODEL_VERSION).inc()

    if random.random() < 0.01:
        logger.info(
            f"prediction={prediction} "
            f"version={MODEL_VERSION} "
            f"amount={amount}"
        )

    return {
        "prediction": prediction,
        "model_version": MODEL_VERSION
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )