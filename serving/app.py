import os
import logging
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="FraudGuard API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-api")

MODEL_VERSION = os.getenv("MODEL_VERSION", "stable")

logger.info(f"Starting FraudGuard API | version={MODEL_VERSION}")


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


prediction_latency = Histogram(
    "fraud_inference_latency_seconds",
    "Model inference latency",
    ["version"]
)


class PredictionRequest(BaseModel):
    amount: float
    merchant: str
    country: str
    user_id: str


class PredictionResponse(BaseModel):
    prediction: int
    model_version: str


def predict_fraud(data: PredictionRequest) -> int:
    """
    Dummy logic for demo / AB testing.
    Replace with real model inference.
    """
    if data.amount > 1000:
        return 1
    return 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.get("/version")
def version():
    return {
        "model": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with prediction_latency.labels(version=MODEL_VERSION).time():

        prediction = predict_fraud(request)

 
        predictions_total.labels(version=MODEL_VERSION).inc()

        if prediction == 1:
            fraud_positive_total.labels(version=MODEL_VERSION).inc()
        else:
            fraud_negative_total.labels(version=MODEL_VERSION).inc()

        logger.info(
            f"prediction={prediction} "
            f"version={MODEL_VERSION} "
            f"amount={request.amount}"
        )

        return PredictionResponse(
            prediction=prediction,
            model_version=MODEL_VERSION
        )


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
