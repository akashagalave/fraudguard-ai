import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.redis_client import init_redis, get_from_cache, set_to_cache
from src.serving.config import FRAUD_THRESHOLD, REDIS_TTL_SECONDS
from src.features.feature_engineering import build_features
from src.serving.shap_utils import generate_shap_bar_chart, generate_human_explanations
from src.serving.email_service import send_fraud_email


# --------------------
# Setup
# --------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="FraudGuard AI", version="2.0.0")

feature_columns = None
artifacts = {}


# --------------------
# Seldon Configuration
# --------------------
SELDON_URL ="http://fraudguard-default.fraudguard.svc.cluster.local:8000/api/v1.0/predictions"



def call_seldon(df: pd.DataFrame) -> float:
    """
    Call Seldon inference service.
    Returns fraud probability.
    """
    payload = {
        "data": {
            "ndarray": df.values.tolist()
        }
    }

    try:
        resp = requests.post(SELDON_URL, json=payload, timeout=2)
        resp.raise_for_status()
        return float(resp.json()["data"]["ndarray"][0])
    except Exception as e:
        logger.exception("Seldon inference failed")
        raise RuntimeError("Model inference service unavailable") from e


# --------------------
# Prometheus Metrics
# --------------------
REQUESTS_TOTAL = Counter(
    "fraud_requests_total",
    "Total fraud prediction requests"
)

FRAUD_PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total fraud predictions"
)

EMAIL_ALERTS_SENT = Counter(
    "fraud_alerts_sent_total",
    "Total fraud email alerts sent"
)

INFERENCE_LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Fraud inference latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)


# --------------------
# Startup
# --------------------
@app.on_event("startup")
async def startup_event():
    global feature_columns, artifacts

    artifacts_path = Path("models/fraudguard_lightgbm/artifacts.pkl")
    artifacts = joblib.load(artifacts_path) if artifacts_path.exists() else {}

    feature_columns = artifacts.get("feature_columns")

    try:
        await init_redis()
    except Exception:
        logger.warning("Redis not available – running without cache")

    logger.info("🚀 FraudGuard API (Seldon-backed) started")


# --------------------
# Helpers
# --------------------
def build_prediction_cache_key(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True)
    return "fraud_pred:" + hashlib.sha256(serialized.encode()).hexdigest()


def build_alert_cache_key(transaction_id: str) -> str:
    return f"fraud_alert_sent:{transaction_id}"


# --------------------
# Metrics Endpoint
# --------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# --------------------
# Prediction API
# --------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    REQUESTS_TOTAL.inc()
    start_time = time.time()

    try:
        pred_key = build_prediction_cache_key(request.features)
        alert_key = build_alert_cache_key(request.transaction_id)

        # --------------------
        # Cache lookup
        # --------------------
        cached = await get_from_cache(pred_key)
        if cached:
            response = cached
        else:
            # --------------------
            # Feature engineering
            # --------------------
            raw_df = pd.DataFrame([request.features])

            engineered_df, _ = build_features(
                raw_df,
                mode="test",
                artifacts=artifacts
            )

            row = {}
            for col in feature_columns:
                val = engineered_df[col].iloc[0] if col in engineered_df else 0
                row[col] = val if isinstance(val, (int, float, np.number)) else 0

            df_final = (
                pd.DataFrame([row])
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )[feature_columns]

            # --------------------
            # 🔥 Seldon Inference
            # --------------------
            prob = call_seldon(df_final)
            is_fraud = prob >= FRAUD_THRESHOLD

            shap_reasons = []
            human_explanations = []

            if is_fraud:
                FRAUD_PREDICTIONS_TOTAL.inc()

                _, shap_reasons = generate_shap_bar_chart(
                    artifacts["explainer"],
                    None,
                    df_final,
                    feature_columns
                )

                human_explanations = generate_human_explanations(shap_reasons)

            response = {
                "fraud_probability": round(prob, 6),
                "risk_scores": int(prob * 100),
                "is_fraud": bool(is_fraud),
                "top_reasons": shap_reasons,
                "human_explanations": human_explanations
            }

            await set_to_cache(pred_key, response, REDIS_TTL_SECONDS)

        # --------------------
        # Email alert (async + idempotent)
        # --------------------
        if response["is_fraud"]:
            already_sent = await get_from_cache(alert_key)

            if not already_sent:
                EMAIL_ALERTS_SENT.inc()

                background_tasks.add_task(
                    send_fraud_email,
                    request.transaction_id,
                    request.features,
                    response["fraud_probability"],
                    response["risk_scores"],
                    None,
                    response["top_reasons"],
                    response["human_explanations"]
                )

                await set_to_cache(alert_key, True, 86400)

        return response

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        INFERENCE_LATENCY.observe(time.time() - start_time)
