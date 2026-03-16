
import json
import logging
import os
import time
import joblib
import pandas as pd
from pathlib import Path

from confluent_kafka import Consumer, Producer, KafkaError
from src.features.feature_engineering import build_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fraudguard-consumer")


KAFKA_BOOTSTRAP = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092",
)
CONSUME_TOPIC = os.getenv("KAFKA_CONSUME_TOPIC", "transactions")
SCORES_TOPIC  = os.getenv("KAFKA_SCORES_TOPIC",  "fraud-scores")
ALERTS_TOPIC  = os.getenv("KAFKA_ALERTS_TOPIC",  "fraud-alerts")
DLQ_TOPIC     = os.getenv("KAFKA_DLQ_TOPIC",     "fraud-dlq")
GROUP_ID      = os.getenv("KAFKA_GROUP_ID",      "fraud-consumer-group")

MODEL_DIR      = Path(os.getenv("MODEL_DIR", "models/fraudguard_lightgbm"))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.3"))
MODEL_VERSION   = os.getenv("MODEL_VERSION", "stable")


def _load_assets():
    logger.info("Loading model assets...")
    model = joblib.load(MODEL_DIR / "model.pkl")

    feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
    categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))

    threshold_path = MODEL_DIR / "threshold.json"
    threshold = (
        float(json.load(open(threshold_path)).get("threshold", FRAUD_THRESHOLD))
        if threshold_path.exists()
        else FRAUD_THRESHOLD
    )


    artifacts_path = MODEL_DIR / "feature_artifacts.pkl"
    artifacts = joblib.load(artifacts_path) if artifacts_path.exists() else None

    logger.info(f"Model loaded | threshold={threshold:.4f}")
    return model, feature_columns, categorical_cols, threshold, artifacts


def _build_feature_df(
    raw: dict,
    feature_columns: list,
    categorical_cols: list,
    artifacts,
) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df.columns = (
        df.columns
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    df, _ = build_features(df, mode="test", artifacts=artifacts)

    # Align columns to training set
    for col in set(feature_columns) - set(df.columns):
        df[col] = 0
    df = df[feature_columns]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _make_consumer():
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": "latest",          
        "enable.auto.commit": True,
        "session.timeout.ms": 30000,
        "heartbeat.interval.ms": 10000,
    })


def _make_producer():
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "linger.ms": 2,
        "acks": "1",
        "retries": 3,
        "socket.timeout.ms": 1000,
    })


def _publish(producer: Producer, topic: str, payload: dict):
    producer.produce(topic, value=json.dumps(payload).encode("utf-8"))
    producer.poll(0)   # non-blocking flush of internal queue



def main():
    model, feature_columns, categorical_cols, threshold, artifacts = _load_assets()

    consumer = _make_consumer()
    producer = _make_producer()
    consumer.subscribe([CONSUME_TOPIC])

    logger.info(f"Consumer started | topic={CONSUME_TOPIC} | group={GROUP_ID}")

    processed = 0
    errors = 0

    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            logger.error(f"Kafka error: {msg.error()}")
            continue

        raw_value = msg.value().decode("utf-8")

        try:
            txn = json.loads(raw_value)
            transaction_id = txn.get("transaction_id", "unknown")


            t0 = time.perf_counter()
            feature_df = _build_feature_df(txn, feature_columns, categorical_cols, artifacts)


            fraud_probability = float(model.predict_proba(feature_df)[0][1])
            is_fraud = fraud_probability >= threshold
            inference_ms = (time.perf_counter() - t0) * 1000

            _publish(producer, SCORES_TOPIC, {
                "transaction_id": transaction_id,
                "fraud_probability": fraud_probability,
                "is_fraud": is_fraud,
                "risk_score": int(fraud_probability * 100),
                "model_version": MODEL_VERSION,
                "inference_latency_ms": round(inference_ms, 2),
            })

            if fraud_probability >= 0.8:
                _publish(producer, ALERTS_TOPIC, {
                    **txn,
                    "fraud_probability": fraud_probability,
                    "model_version": MODEL_VERSION,
                    "event_type": "FRAUD_DETECTED",
                })

            processed += 1

 
            if processed % 1000 == 0:
                logger.info(
                    f"Processed {processed} transactions | "
                    f"errors={errors} | "
                    f"last_latency_ms={inference_ms:.1f}"
                )

        except Exception as e:
            errors += 1
            logger.error(f"Processing failed for message: {e}")

            try:
                _publish(producer, DLQ_TOPIC, {
                    "error": str(e),
                    "original_message": raw_value,
                    "topic": CONSUME_TOPIC,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                })
            except Exception as dlq_err:
                logger.critical(f"DLQ publish also failed: {dlq_err}")


if __name__ == "__main__":
    main()
