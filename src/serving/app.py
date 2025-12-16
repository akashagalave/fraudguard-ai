import hashlib
import json
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

from src.serving.schemas import PredictionRequest, PredictionResponse, ShapReason
from src.serving.model_loader import load_artifacts
from src.serving.redis_client import get_from_cache, set_to_cache
from src.serving.config import FRAUD_THRESHOLD, REDIS_TTL_SECONDS

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

app = FastAPI(title="FraudGuard AI", version="1.0.0")

model, explainer, feature_columns, categorical_cols = load_artifacts()


def build_cache_key(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True)
    return "fraud_pred:" + hashlib.sha256(serialized.encode()).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        cache_key = build_cache_key(request.features)

        cached = get_from_cache(cache_key)
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

        shap_values = explainer.shap_values(df)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  

        shap_row = shap_values[0]
        shap_abs = np.abs(shap_row)

        top_idx = np.argsort(shap_abs)[-5:][::-1]

        top_reasons_models = [
            ShapReason(
                feature=feature_columns[i],
                impact=round(float(shap_row[i]), 6)
        )
        for i in top_idx
     ] 
        top_reasons_dict = [r.model_dump() for r in top_reasons_models]
        response = {
        "fraud_probability": round(prob, 6),
        "is_fraud": bool(is_fraud),
        "top_reasons": top_reasons_dict
        }
        
        set_to_cache(cache_key, response, REDIS_TTL_SECONDS)
        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
