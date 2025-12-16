import hashlib
import json
import logging
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.model_loader import load_model_assets
from src.serving.redis_client import (
    init_redis,
    get_from_cache,
    set_to_cache
)
from src.serving.config import FRAUD_THRESHOLD, REDIS_TTL_SECONDS

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

app = FastAPI(title="FraudGuard AI", version="1.0.0")

model = None
feature_columns = None
categorical_cols = None


@app.on_event("startup")
async def startup_event():
    global model, feature_columns, categorical_cols
    model, feature_columns, categorical_cols = load_model_assets()
    await init_redis()


def build_cache_key(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True)
    return "fraud_pred:" + hashlib.sha256(serialized.encode()).hexdigest()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        cache_key = build_cache_key(request.features)
        cached = await get_from_cache(cache_key)
        if cached:
            logger.info("Cache HIT")
            return cached

        logger.info("Cache MISS")

        row = {col: None for col in feature_columns}
        for k, v in request.features.items():
            if k in row:
                row[k] = v

        df = pd.DataFrame([row])

        for col in categorical_cols:
            df[col] = df[col].astype("category")

        numeric_cols = [c for c in feature_columns if c not in categorical_cols]
        df[numeric_cols] = df[numeric_cols].fillna(0)

        prob = float(model.predict_proba(df)[0][1])
        is_fraud = prob >= FRAUD_THRESHOLD

        response = {
            "fraud_probability": round(prob, 6),
            "is_fraud": bool(is_fraud),
            "top_reasons": None
        }
        await set_to_cache(cache_key, response, REDIS_TTL_SECONDS)

        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
