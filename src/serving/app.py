import json
import logging
import time
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.redis_client import init_redis
from src.features.feature_engineering import build_features


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-api")

app = FastAPI(title="FraudGuard AI", version="FINAL-STABLE")

SELDON_URL = os.getenv(
    "SELDON_URL",
    "http://fraudguard-model-default.fraudguard.svc.cluster.local:8000/api/v1.0/predictions"
)

FRAUD_THRESHOLD = 0.3
feature_columns: list[str] | None = None

REQUESTS_TOTAL = Counter("fraud_requests_total", "Total prediction requests")
FRAUD_PREDICTIONS_TOTAL = Counter("fraud_predictions_total", "Fraud predictions")

INFERENCE_LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Prediction latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)


@app.on_event("startup")
async def startup_event():
    global feature_columns

    logger.info("🚀 FraudGuard API starting...")

    try:
        with open("/app/models/feature_columns.json") as f:
            feature_columns = json.load(f)

        logger.info(f" Loaded {len(feature_columns)} feature columns")

    except Exception as e:
        logger.error(f" Failed to load feature_columns.json: {e}")
        raise e

    await init_redis()
    logger.info(f" Seldon target: {SELDON_URL}")
    logger.info(" FraudGuard API ready")

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUESTS_TOTAL.inc()
    start_time = time.time()

    try:
        raw_df = pd.DataFrame([request.features])

        features_df, _ = build_features(
            raw_df,
            mode="test",
            artifacts=None
        )

        features_df = features_df.reindex(
            columns=feature_columns,
            fill_value=0.0
        )

        logger.info(f"📤 Sending to Seldon | Shape={features_df.shape}")

        resp = requests.post(
            SELDON_URL,
            json={"data": {"ndarray": features_df.values.tolist()}},
            timeout=5
        )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Seldon error: {resp.text}"
            )

        seldon_response = resp.json()
        logger.info(f"📥 Raw Seldon response: {seldon_response}")


        prob = None

        if "jsonData" in seldon_response:
            prob = seldon_response["jsonData"]["data"]["ndarray"][0]

        elif "data" in seldon_response:
            prob = seldon_response["data"]["ndarray"][0]

        elif "predictions" in seldon_response:
            pred = seldon_response["predictions"][0]
            if isinstance(pred, dict):
                prob = pred["data"]["ndarray"][0]
            else:
                prob = pred

        if prob is None:
            raise ValueError(f"Unsupported Seldon response: {seldon_response}")

        prob = float(prob)

        risk_score = int(round(prob * 100))
        is_fraud = prob >= FRAUD_THRESHOLD

        if is_fraud:
            FRAUD_PREDICTIONS_TOTAL.inc()

        return {
            "fraud_probability": round(prob, 6),
            "risk_score": risk_score,
            "is_fraud": is_fraud,
            "model_version": "lightgbm-v1",
            "top_reasons": None,
            "human_explanations": None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_LATENCY.observe(time.time() - start_time)
