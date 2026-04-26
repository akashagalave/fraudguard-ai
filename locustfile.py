
import random
import time
import uuid
from locust import HttpUser, task, between

MERCHANTS = [
    "fraud_Kerluke-Abshire",
    "amazon_store",
    "flipkart_retail",
    "zomato_food",
    "hdfc_bank",
]

CATEGORIES = [
    "shopping_net",
    "shopping_pos",
    "food_dining",
    "gas_transport",
    "misc_net",
]

JOBS = ["Engineer", "Doctor", "Teacher", "Unemployed", "Student"]

def normal_payload():
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "features": {
            "trans_date_trans_time": "2024-06-15 14:30:00",
            "cc_num": f"4{random.randint(100000000000000, 999999999999999)}",
            "merchant": random.choice(MERCHANTS[1:]),
            "category": random.choice(CATEGORIES),
            "amt": round(random.uniform(50, 2000), 2),
            "gender": random.choice(["M", "F"]),
            "lat": 19.076,
            "long": 72.8777,
            "merch_lat": 19.10,
            "merch_long": 72.90,
            "city_pop": 1500000,
            "job": random.choice(JOBS),
            "dob": "1995-05-15",
            "unix_time": int(time.time()),
        }
    }

def fraud_payload():
    return {
        "transaction_id": f"txn_fraud_{uuid.uuid4().hex[:12]}",
        "features": {
            "trans_date_trans_time": "2024-06-15 02:30:00",
            "cc_num": f"4{random.randint(100000000000000, 999999999999999)}",
            "merchant": "fraud_Kerluke-Abshire",
            "category": "misc_net",
            "amt": round(random.uniform(8000, 50000), 2),
            "gender": random.choice(["M", "F"]),
            "lat": 19.076,
            "long": 72.8777,
            "merch_lat": 40.7128,
            "merch_long": -74.0060,
            "city_pop": 300,
            "job": "Unemployed",
            "dob": "1995-05-15",
            "unix_time": int(time.time()),
        }
    }


class FraudGuardUser(HttpUser):
    wait_time = between(0, 0)

    @task(9)
    def predict_normal(self):
        self.client.post(
            "/predict",
            json=normal_payload(),
            name="/predict [normal]"
        )

    @task(1)
    def predict_fraud(self):
        self.client.post(
            "/predict",
            json=fraud_payload(),
            name="/predict [fraud]"
        )


