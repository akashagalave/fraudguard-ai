"""
SHAP Worker for FraudGuard

Responsibilities:
- Consume fraud-events from Kafka
- Compute per-transaction SHAP explanations
"""

import json
import logging
import os
import time
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import shap
from confluent_kafka import Consumer

# 🔑 Reuse training preprocessing
from src.features.feature_engineering import build_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-shap-worker")

# ============================
# Config
# ============================
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092",
)

KAFKA_TOPIC = "fraud-events"
KAFKA_GROUP_ID = "shap-worker-group"

MODEL_PATH = "/app/model.pkl"
FEATURE_COLUMNS_PATH = "/app/feature_columns.json"
ARTIFACTS_PATH = "/app/models/fraudguard_lightgbm"

TOP_K = 5

# ============================
# Load Model & SHAP Explainer
# ============================
logger.info("📥 Loading model and SHAP explainer...")

model = joblib.load(MODEL_PATH)

with open(FEATURE_COLUMNS_PATH) as f:
    feature_columns: List[str] = json.load(f)

explainer = shap.TreeExplainer(model)

logger.info("✅ Model & SHAP explainer loaded")

# ============================
# Kafka Consumer
# ============================
def create_consumer() -> Consumer:
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    }
    return Consumer(conf)

# ============================
# SHAP Logic
# ============================
def compute_shap(features: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Compute SHAP values using the SAME preprocessing as training.
    """

    raw_df = pd.DataFrame([features])

    # 🔥 Reuse training preprocessing
    features_df, _ = build_features(
        raw_df,
        mode="test",
        artifacts=ARTIFACTS_PATH
    )

    # 🔒 Force feature alignment
    features_df = features_df.reindex(columns=feature_columns, fill_value=0.0)

    shap_values = explainer.shap_values(features_df)

    # Fraud class = index 1
    values = shap_values[1][0]

    shap_pairs = list(zip(feature_columns, values))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": feature, "impact": float(value)}
        for feature, value in shap_pairs[:TOP_K]
    ]

# ============================
# Main Loop
# ============================
def main():
    logger.info("🚀 SHAP Worker starting...")

    consumer = create_consumer()
    consumer.subscribe([KAFKA_TOPIC])

    logger.info(f"📥 Subscribed to Kafka topic: {KAFKA_TOPIC}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))

            logger.info(
                f"🧠 Computing SHAP for txn={event['transaction_id']}"
            )

            top_reasons = compute_shap(event["features"])

            logger.info(
                f"✅ SHAP completed | txn={event['transaction_id']} | reasons={top_reasons}"
            )

            time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("🛑 SHAP Worker shutting down")

    finally:
        consumer.close()

if __name__ == "__main__":
    main()
