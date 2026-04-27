import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any

from confluent_kafka import Producer

logger = logging.getLogger("fraudguard-kafka")


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092",
)

FRAUD_TOPIC = "fraud-events"
DLQ_TOPIC = "fraud-events-dlq"

_producer: Producer | None = None

def init_kafka_producer() -> None:
    global _producer

    try:
        conf = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "linger.ms": 5,
            "acks": "all",  
            "retries": 3,
            "socket.timeout.ms": 1000,
        }

        _producer = Producer(conf)
        logger.info(" Kafka producer initialized")

    except Exception:
        logger.error(" Failed to initialize Kafka producer", exc_info=True)
        _producer = None


def _delivery_report(err, msg):
    if err is not None:
        logger.error(f" Delivery failed: {err}")
    else:
        logger.debug(
            f" Delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}"
        )


def build_fraud_event(
    *,
    transaction_id: str,
    user_email: str,
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
        "user_email": user_email,
    }


def send_fraud_event(event: Dict[str, Any], retries: int = 3) -> None:
    if _producer is None:
        logger.warning("Kafka producer not available, skipping event")
        return

    payload = json.dumps(event).encode("utf-8")

    for attempt in range(retries):
        try:
            _producer.produce(
                topic=FRAUD_TOPIC,
                value=payload,
                callback=_delivery_report,
            )
            _producer.poll(0)
            return

        except Exception as e:
            logger.error(
                f" Kafka send failed (attempt {attempt+1}/{retries}): {e}",
                exc_info=True,
            )
            time.sleep(2 ** attempt) 

  
    try:
        logger.error(" Sending event to DLQ after retries exhausted")

        _producer.produce(
            topic=DLQ_TOPIC,
            value=payload,
            callback=_delivery_report,
        )
        _producer.poll(0)

    except Exception:
        logger.critical(" DLQ send failed — DATA LOSS POSSIBLE", exc_info=True)



def flush_producer():
    if _producer is not None:
        _producer.flush(timeout=5)