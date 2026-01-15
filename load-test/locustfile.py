from locust import HttpUser, task, between

class FraudUser(HttpUser):
    wait_time = between(0, 0)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "amount": 5000,
                "merchant": "test",
                "country": "IN",
                "user_id": "u1"
            }
        )