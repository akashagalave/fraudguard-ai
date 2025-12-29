import json
import requests
from kafka import KafkaConsumer

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"
FASTAPI_URL = "http://localhost:8000/predict"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-consumer-group"
)

print("Kafka Consumer started...")

for message in consumer:
    event = message.value
    try:
        resp = requests.post(FASTAPI_URL, json=event, timeout=5)
        print(
            f"Txn {event['transaction_id']} → "
            f"Status {resp.status_code} | Response: {resp.json()}"
        )
    except Exception as e:
        print(f"Failed processing txn {event['transaction_id']}: {e}")
