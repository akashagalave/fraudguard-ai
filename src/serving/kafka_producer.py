
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger("fraudguard-kafka")


FRAUD_TOPIC = "fraud-events"


def build_fraud_event(
    *,
    transaction_id: str,
    fraud_probability: float,
    risk_score: int,
    model_version: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a fraud event payload to be sent to Kafka.

    This is the SINGLE SOURCE OF TRUTH for the event schema.
    """

    return {
        "event_type": "FRAUD_DETECTED",
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "risk_score": risk_score,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,  
    }


def serialize_event(event: Dict[str, Any]) -> bytes:
    """
    Serialize event for Kafka transport.
    """
    return json.dumps(event).encode("utf-8")


