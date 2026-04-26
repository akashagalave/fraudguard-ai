import os
import time
import json
import logging
import random
import joblib
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .model_loader import load_model_assets
from .schemas import PredictionRequest, PredictionResponse, ShapReason
from .config import MODEL_VERSION, FRAUD_THRESHOLD, REDIS_TTL_SECONDS
from .redis_client import init_redis, get_from_cache, set_to_cache
from .kafka_producer import init_kafka_producer, send_fraud_event, build_fraud_event
from src.features.feature_engineering import build_features

app = FastAPI(title="FraudGuard API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-api")
logger.info(f"Starting FraudGuard API | version={MODEL_VERSION}")


_INFERENCE_BUCKETS = (
    0.003, 0.005, 0.008, 0.010, 0.015,
    0.020, 0.025, 0.035, 0.050, 0.100,
    0.250, 0.500, 1.0, float("inf"),
)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_latency = Histogram(
    "http_request_latency_seconds",
    "End-to-end application latency (FastAPI)",
    ["method", "path"],
    buckets=_INFERENCE_BUCKETS,
)

model_inference_latency = Histogram(
    "fraud_model_inference_latency_seconds",
    "Pure model inference latency",
    ["version"],
    buckets=_INFERENCE_BUCKETS,
)

predictions_total = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["version", "is_fraud"],  
)

fraud_positive_total = Counter(
    "fraud_positive_predictions_total",
    "Fraud positive predictions",
    ["version"],
)

fraud_negative_total = Counter(
    "fraud_negative_predictions_total",
    "Fraud negative predictions",
    ["version"],
)

fraud_probability_histogram = Histogram(
    "fraud_probability_distribution",
    "Distribution of raw fraud probability scores",
    ["version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


_model = None
_feature_columns = None
_categorical_cols = None
_feature_artifacts = None   
_shap_explainer = None
_threshold = FRAUD_THRESHOLD  


@app.on_event("startup")
async def startup():
    global _model, _feature_columns, _categorical_cols, \
           _feature_artifacts, _shap_explainer, _threshold

    logger.info("Loading model assets...")
    _model, _feature_columns, _categorical_cols = load_model_assets()

   
    artifacts_path = Path("models/fraudguard_lightgbm/feature_artifacts.pkl")
    if artifacts_path.exists():
        _feature_artifacts = joblib.load(artifacts_path)
        logger.info("Feature artifacts loaded")
    else:
        logger.warning("feature_artifacts.pkl not found — using fallback FE (no card/merchant stats)")
        _feature_artifacts = None

   
    threshold_path = Path("models/fraudguard_lightgbm/threshold.json")
    if threshold_path.exists():
        saved = json.load(open(threshold_path))
        _threshold = float(saved.get("threshold", FRAUD_THRESHOLD))
        logger.info(f"Optimal threshold loaded: {_threshold:.4f}")
    else:
        logger.info(f"Using config threshold: {_threshold}")

   
    await init_redis()

    init_kafka_producer()

    logger.info(f"FraudGuard API ready | version={MODEL_VERSION} | threshold={_threshold:.4f}")



@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    http_requests_total.labels(
        method=request.method,
        path=request.url.path,
        status=response.status_code,
    ).inc()

    http_request_latency.labels(
        method=request.method,
        path=request.url.path,
    ).observe(duration)

    return response



def _build_feature_df(features: dict) -> pd.DataFrame:

    df = pd.DataFrame([features])


    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    df, _ = build_features(df, mode="test", artifacts=_feature_artifacts)

   
    missing = set(_feature_columns) - set(df.columns)
    for col in missing:
        df[col] = 0

    df = df[_feature_columns]

  
    for col in _categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

def _emit_kafka_event(
    transaction_id: str,
    features: dict,
    fraud_probability: float,
    risk_score: int,
):

    try:
        event = build_fraud_event(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            risk_score=risk_score,
            model_version=MODEL_VERSION,
            features=features,
        )
        send_fraud_event(event)
    except Exception as e:
        logger.error(f"Kafka emit failed for txn {transaction_id}: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):

    transaction_id = request.transaction_id
    features_dict = request.features

 
    cache_key = f"fraud:{transaction_id}"
    cached = await get_from_cache(cache_key)
    if cached:
        logger.debug(f"Cache hit for txn {transaction_id}")
        return PredictionResponse(**cached)


    try:
        feature_df = _build_feature_df(features_dict)
    except Exception as e:
        logger.error(f"Feature engineering failed for txn {transaction_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {str(e)}"
        )

    with model_inference_latency.labels(version=MODEL_VERSION).time():
        fraud_probability = float(_model.predict_proba(feature_df)[0][1])

    is_fraud = fraud_probability >= _threshold
    risk_score = int(fraud_probability * 100)


    predictions_total.labels(
        version=MODEL_VERSION,
        is_fraud=str(is_fraud),
    ).inc()

    if is_fraud:
        fraud_positive_total.labels(version=MODEL_VERSION).inc()
    else:
        fraud_negative_total.labels(version=MODEL_VERSION).inc()

    fraud_probability_histogram.labels(version=MODEL_VERSION).observe(fraud_probability)

    if random.random() < 0.01:
        logger.info(
            f"txn={transaction_id} "
            f"prob={fraud_probability:.4f} "
            f"is_fraud={is_fraud} "
            f"version={MODEL_VERSION}"
        )

    result = PredictionResponse(
        fraud_probability=fraud_probability,
        risk_score=risk_score,
        is_fraud=is_fraud,
        model_version=MODEL_VERSION,
    )

    await set_to_cache(cache_key, result.model_dump(), ttl=REDIS_TTL_SECONDS)

    if is_fraud:
        background_tasks.add_task(
            _emit_kafka_event,
            transaction_id,
            features_dict,
            fraud_probability,
            risk_score,
        )

    return result


@app.post("/predict/explain", response_model=PredictionResponse)
async def predict_with_explain(request: PredictionRequest):
    """
    Fraud inference with SHAP explanations.
    Slower than /predict — use for investigation/audit, not real-time path.
    """
    import shap as shap_lib
    import numpy as np

    transaction_id = request.transaction_id
    features_dict = request.features

    try:
        feature_df = _build_feature_df(features_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {str(e)}")

    with model_inference_latency.labels(version=MODEL_VERSION).time():
        fraud_probability = float(_model.predict_proba(feature_df)[0][1])

    is_fraud = fraud_probability >= _threshold

    # SHAP explanation
    shap_reasons = None
    if _shap_explainer is not None:
        try:
            shap_values = _shap_explainer.shap_values(feature_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            sv = shap_values[0]
            top_idx = np.argsort(np.abs(sv))[-5:][::-1]
            shap_reasons = [
                ShapReason(feature=_feature_columns[i], impact=float(sv[i]))
                for i in top_idx
            ]
        except Exception as e:
            logger.warning(f"SHAP failed for txn {transaction_id}: {e}")

    predictions_total.labels(version=MODEL_VERSION, is_fraud=str(is_fraud)).inc()

    return PredictionResponse(
        fraud_probability=fraud_probability,
        risk_score=int(fraud_probability * 100),
        is_fraud=is_fraud,
        model_version=MODEL_VERSION,
        top_reasons=shap_reasons,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": _model is not None,
        "threshold": _threshold,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)