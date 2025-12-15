import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from src.serving.model_loader import load_artifacts
from src.serving.config import FRAUD_THRESHOLD

logger = logging.getLogger("fraudguard_api")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(
    title="FraudGuard AI",
    version="1.0.0",
    description="Production-grade fraud detection API"
)


logger.info("Loading model from MLflow Registry")

model, FEATURE_COLUMNS, CATEGORICAL_COLS = load_artifacts()

logger.info("Model loaded successfully")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        row = {col: None for col in FEATURE_COLUMNS}

        for key, value in request.features.items():
            if key in row:
                row[key] = value

        df = pd.DataFrame([row])

        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category")

        numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]
        df[numeric_cols] = df[numeric_cols].fillna(0)

        prob = float(model.predict_proba(df)[0][1])
        is_fraud = prob >= FRAUD_THRESHOLD

        return PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=bool(prob >= FRAUD_THRESHOLD),
            top_reasons=None
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
