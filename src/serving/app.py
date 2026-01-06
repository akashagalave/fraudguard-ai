import hashlib
import json
import logging
import pandas as pd
import shap
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from pathlib import Path

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.model_loader import load_model_assets
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

app = FastAPI(title="FraudGuard AI", version="1.0.0")

model = None
feature_columns = None
artifacts = {}
explainer = None


# --------------------
# Startup
# --------------------
@app.on_event("startup")
async def startup_event():
    global model, feature_columns, artifacts, explainer

    model, feature_columns, _ = load_model_assets()

    artifact_path = Path("models/fraudguard_lightgbm/artifacts.pkl")
    artifacts = joblib.load(artifact_path) if artifact_path.exists() else {}

    explainer = shap.TreeExplainer(model)

    try:
        await init_redis()
    except Exception:
        logger.warning("Redis not available – running without cache")

    logger.info("🚀 App startup completed")


# --------------------
# Helpers
# --------------------
def build_prediction_cache_key(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True)
    return "fraud_pred:" + hashlib.sha256(serialized.encode()).hexdigest()


def build_alert_cache_key(transaction_id: str) -> str:
    return f"fraud_alert_sent:{transaction_id}"


# --------------------
# Prediction API
# --------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    try:
        pred_key = build_prediction_cache_key(request.features)
        alert_key = build_alert_cache_key(request.transaction_id)

        # --------------------
        # 1. Try cache
        # --------------------
        cached = await get_from_cache(pred_key)
        if cached:
            response = cached
        else:
            # --------------------
            # 2. Feature Engineering
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
            # 3. Model inference
            # --------------------
            prob = float(model.predict_proba(df_final)[0][1])
            is_fraud = prob >= FRAUD_THRESHOLD

            shap_img = None
            shap_reasons = []
            human_explanations = []

            if is_fraud:
                shap_img, shap_reasons = generate_shap_bar_chart(
                    explainer, model, df_final, feature_columns
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
        # 4. Fraud Email (idempotent + async)
        # --------------------
        if response["is_fraud"]:
            already_sent = await get_from_cache(alert_key)

            if not already_sent:
                background_tasks.add_task(
                    send_fraud_email,
                    request.transaction_id,
                    request.features,
                    response["fraud_probability"],
                    response["risk_scores"],
                    None,  # SHAP image optional
                    response["top_reasons"],
                    response["human_explanations"]
                )

                await set_to_cache(alert_key, True, 86400)

        return response

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
