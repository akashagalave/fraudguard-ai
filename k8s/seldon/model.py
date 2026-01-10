import joblib
import json
import numpy as np

class Model:
    def __init__(self):
        print("📥 Loading model...")

        self.model = joblib.load("/app/model.pkl")

        with open("/app/feature_columns.json") as f:
            self.feature_columns = json.load(f)

        self.expected_features = len(self.feature_columns)
        print(f" Model loaded | Expecting {self.expected_features} features")

    def predict(self, X, feature_names=None, meta=None):
        X = np.array(X)
        print(f"🔮 Input shape: {X.shape}")

        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"Feature mismatch: expected {self.expected_features}, got {X.shape[1]}"
            )

        prob = float(self.model.predict_proba(X)[0][1])

        return {
            "data": {
                "names": ["probability"],
                "ndarray": [prob]
            }
        }
