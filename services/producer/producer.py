import json
import time
import uuid
import random
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"

SLEEP_SECONDS = 0.01  

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=5,
    retries=3
)

def generate_transaction():
    return {
        "transaction_id": f"txn_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "features": {
            "amt": round(random.uniform(10, 20000), 2),
            "gender": random.choice(["M", "F"]),
            "category": random.choice(["misc_net", "shopping", "food"]),
            "merchant": "fraud_Kerluke-Abshire",
            "job": random.choice(["Engineer", "Unemployed", "Doctor"]),
            "hour": random.randint(0, 23),
            "unix_time": int(time.time()),
            "merchant_txn_count": random.randint(1, 5),
            "time_since_prev": random.randint(1, 1000),
            "weekend_ratio": round(random.uniform(0, 1), 2),
            "age": random.randint(18, 70)
        }
    }

if __name__ == "__main__":
    print("Kafka Producer started | streaming transactions continuously")

    while True:
        event = generate_transaction()
        producer.send(TOPIC, value=event)
        time.sleep(SLEEP_SECONDS)

