import hashlib
import json
import logging
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.model_loader import load_model_assets
from src.serving.redis_client import init_redis, get_from_cache, set_to_cache
from src.serving.config import FRAUD_THRESHOLD, REDIS_TTL_SECONDS
from src.serving.shap_utils import (
    generate_shap_bar_chart,
    generate_human_explanations
)
from src.serving.email_service import send_fraud_email

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="FraudGuard AI", version="1.0.0")

model = None
explainer = None
feature_columns = None
categorical_cols = None


@app.on_event("startup")
async def startup_event():
    global model, explainer, feature_columns, categorical_cols

    model, feature_columns, categorical_cols = load_model_assets()
    explainer = shap.TreeExplainer(model)

    await init_redis()
    logger.info("App startup completed")


def build_prediction_cache_key(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True)
    return "fraud_pred:" + hashlib.sha256(serialized.encode()).hexdigest()


def build_alert_cache_key(transaction_id: str) -> str:
    return f"fraud_alert_sent:{transaction_id}"


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        pred_key = build_prediction_cache_key(request.features)
        alert_key = build_alert_cache_key(request.transaction_id)

        shap_img = None
        cached = await get_from_cache(pred_key)

        if cached:
            logger.info("Prediction Cache HIT")
            response = cached
        else:
            logger.info("Prediction Cache MISS")

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
                "top_reasons": [],
                "human_explanations": []
            }

            if is_fraud:
                shap_img, shap_pairs = generate_shap_bar_chart(
                    explainer, model, df, feature_columns
                )

                shap_structured = [
                    {"feature": f, "impact": float(i)}
                    for f, i in shap_pairs
                ]

                response["top_reasons"] = shap_structured
                response["human_explanations"] = generate_human_explanations(
                    shap_structured
                )

            await set_to_cache(pred_key, response, REDIS_TTL_SECONDS)

        if response["is_fraud"]:
            if not await get_from_cache(alert_key):
                logger.info(
                    f"New fraud event → sending email | txn_id={request.transaction_id}"
                )

                send_fraud_email(
                    transaction_id=request.transaction_id,
                    features=request.features,
                    fraud_probability=response["fraud_probability"],
                    shap_image=shap_img,
                    shap_reasons=response["top_reasons"],
                    human_explanations=response["human_explanations"]
                )

                await set_to_cache(alert_key, True, 24 * 60 * 60)
            else:
                logger.info(
                    f"Fraud email already sent | txn_id={request.transaction_id}"
                )

        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
