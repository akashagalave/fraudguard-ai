import json
import logging
import os
import time
from confluent_kafka import Consumer, Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudguard-shap-worker")

KAFKA_BOOTSTRAP = "fraudguard-kafka-kafka-bootstrap.fraudguard.svc.cluster.local:9092"
CONSUME_TOPIC = "fraud-events"
PRODUCE_TOPIC = "fraud-alerts"
GROUP_ID = "fraudguard-shap-worker"

consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": GROUP_ID,
    "auto.offset.reset": "earliest"
})

producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP
})

def main():
    logger.info(" SHAP Worker starting...")
    consumer.subscribe([CONSUME_TOPIC])

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logger.error(msg.error())
            continue

        event = json.loads(msg.value().decode())

        if event.get("event_type") != "FRAUD_DETECTED":
            continue

     
        alert_event = {
            **event,
            "event_type": "FRAUD_ALERT_READY"
        }

        producer.produce(
            PRODUCE_TOPIC,
            value=json.dumps(alert_event)
        )
        producer.flush()

        logger.info(f" Alert published for txn={event['transaction_id']}")

if __name__ == "__main__":
    main()
