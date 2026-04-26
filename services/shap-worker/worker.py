import json
import logging
import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from confluent_kafka import Consumer
import shap

sys.path.append("/app")
from serving.email_service import send_fraud_email
from shap_utils import generate_shap_bar_chart, generate_human_explanations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fraudguard-shap-worker")

KAFKA_BOOTSTRAP = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092",
)
CONSUME_TOPIC = "fraud-events"
GROUP_ID      = "fraudguard-shap-worker"

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/fraudguard_lightgbm"))

logger.info("Model load ho raha hai...")
model           = joblib.load(MODEL_DIR / "model.pkl")
feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
explainer       = shap.TreeExplainer(model)
logger.info("Model + SHAP explainer ready")

consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id":          GROUP_ID,
    "auto.offset.reset": "earliest",
})


def build_feature_df(features: dict) -> pd.DataFrame:
    df = pd.DataFrame([features])

    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace(r"_+",            "_", regex=True)
        .str.strip("_")
    )

   
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


def handle_fraud_event(event: dict):

    transaction_id   = event.get("transaction_id", "unknown")
    features         = event.get("features", {})
    fraud_probability = event.get("fraud_probability", 0.0)
    risk_score       = event.get("risk_score", 0)

    logger.info(f"Processing fraud event | txn={transaction_id}")

    feature_df = build_feature_df(features)

    shap_image_b64, shap_reasons = generate_shap_bar_chart(
        explainer=explainer,
        model=model,
        df=feature_df,
        feature_columns=feature_columns,
        top_k=5,
    )

    human_explanations = generate_human_explanations(shap_reasons, max_lines=3)

    logger.info(
        f"SHAP done | txn={transaction_id} | "
        f"top={shap_reasons[0]['feature'] if shap_reasons else 'n/a'}"
    )

    send_fraud_email(
        transaction_id=transaction_id,
        features=features,
        fraud_probability=fraud_probability,
        risk_scores=risk_score,
        shap_image=shap_image_b64,
        shap_reasons=shap_reasons,
        human_explanations=human_explanations,
    )

    logger.info(f"Email sent | txn={transaction_id}")


def main():
    logger.info("SHAP Worker start ho raha hai...")
    consumer.subscribe([CONSUME_TOPIC])

    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            continue

        try:
            event = json.loads(msg.value().decode())

            if event.get("event_type") != "FRAUD_DETECTED":
                continue

            handle_fraud_event(event)

        except Exception as e:
          
            logger.error(f"Event processing failed: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    main()