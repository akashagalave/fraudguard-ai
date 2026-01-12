import json
import logging
import os
from confluent_kafka import Consumer
from prometheus_client import Counter, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-alert-worker")


FRAUD_ALERTS_TOTAL = Counter(
    "fraud_alerts_total",
    "Total fraud alerts generated"
)


KAFKA_BOOTSTRAP = "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092"
TOPIC = "fraud-alerts"
GROUP_ID = "fraudguard-alert-worker"

consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": GROUP_ID,
    "auto.offset.reset": "latest"
})


def main():
 
    start_http_server(8001)
    logger.info(" Metrics exposed on :8001/metrics")

    logger.info(" Alert Worker starting (EMAIL DISABLED)")
    consumer.subscribe([TOPIC])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logger.error(msg.error())
            continue

        event = json.loads(msg.value().decode("utf-8"))

        if event.get("event_type") != "FRAUD_ALERT_READY":
            continue

        FRAUD_ALERTS_TOTAL.inc()
        logger.warning(" FRAUD ALERT GENERATED | txn=%s", event["transaction_id"])

if __name__ == "__main__":
    main()
