import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudGuardModel:
    def __init__(self):
        """
        Called once when container starts
        """
        model_path = "/models/model.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        logger.info("✅ FraudGuard model loaded successfully")

    def predict(self, X, names=None):
        """
        Seldon inference entrypoint
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("Input must be 2D array")

        probs = self.model.predict_proba(X)[:, 1]
        return probs.reshape(-1, 1)
