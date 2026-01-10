"""
Kafka producer for FraudGuard (fire-and-forget).

Rules:
- Must NEVER block inference
- Failures must be logged, not raised
- Used ONLY when fraud is detected
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any

from confluent_kafka import Producer

logger = logging.getLogger("fraudguard-kafka")

# ============================
# Config
# ============================
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092",
)

FRAUD_TOPIC = "fraud-events"

_producer: Producer | None = None


# ============================
# Kafka Producer Init
# ============================
def init_kafka_producer() -> None:
    global _producer

    try:
        conf = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "linger.ms": 5,            # small batching
            "acks": "1",               # fast, not blocking
            "retries": 3,
            "socket.timeout.ms": 1000,
        }

        _producer = Producer(conf)
        logger.info("✅ Kafka producer initialized")

    except Exception as e:
        logger.error("❌ Failed to initialize Kafka producer", exc_info=True)
        _producer = None


# ============================
# Event Builder
# ============================
def build_fraud_event(
    *,
    transaction_id: str,
    fraud_probability: float,
    risk_score: int,
    model_version: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "event_type": "FRAUD_DETECTED",
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "risk_score": risk_score,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }


# ============================
# Fire-and-forget send
# ============================
def send_fraud_event(event: Dict[str, Any]) -> None:
    if _producer is None:
        logger.warning("Kafka producer not available, skipping event")
        return

    try:
        _producer.produce(
            topic=FRAUD_TOPIC,
            value=json.dumps(event).encode("utf-8"),
        )
        _producer.poll(0)  # non-blocking

    except Exception:
        logger.error("❌ Failed to send Kafka event", exc_info=True)
