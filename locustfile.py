import random
import time
import uuid
from locust import HttpUser, task, between

# -----------------------------
# CONSTANTS
# -----------------------------
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


# -----------------------------
# PAYLOAD GENERATORS
# -----------------------------
def random_email():
    return f"test_{uuid.uuid4().hex[:6]}@gmail.com"


def normal_payload():
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "user_email": random_email(),  # ✅ REQUIRED
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
        "user_email": random_email(),  # ✅ REQUIRED
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


# -----------------------------
# USER CLASS
# -----------------------------
class FraudGuardUser(HttpUser):
    wait_time = between(0.5, 1.5)  # ✅ realistic load

    @task(8)
    def predict_normal(self):
        payload = normal_payload()

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [normal]",
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                data = response.json()

                # ✅ Validate response
                if "fraud_probability" not in data:
                    response.failure("Missing fraud_probability")

                elif not isinstance(data["fraud_probability"], float):
                    response.failure("Invalid fraud_probability type")

                else:
                    response.success()

            except Exception as e:
                response.failure(f"JSON parse error: {e}")


    @task(2)
    def predict_fraud(self):
        payload = fraud_payload()

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [fraud]",
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                data = response.json()

                if not data.get("is_fraud", False):
                    response.failure("Expected fraud but got normal")

                else:
                    response.success()

            except Exception as e:
                response.failure(f"JSON parse error: {e}")