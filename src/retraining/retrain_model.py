import os
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retraining")

# ============================
# Paths
# ============================
MODEL_REGISTRY = "model"
VERSION_FILE = "model/version.txt"

# ============================
# Helpers
# ============================
def get_next_version():
    if not os.path.exists(VERSION_FILE):
        return "v2"

    with open(VERSION_FILE) as f:
        last = f.read().strip()

    next_version = f"v{int(last.replace('v','')) + 1}"
    return next_version


def save_version(version: str):
    os.makedirs(MODEL_REGISTRY, exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(version)


# ============================
# Retraining
# ============================
def retrain_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "is_fraud"
):
    logger.info("🔁 Retraining model started")

    X = train_df[feature_cols]
    y = train_df[target_col]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    logger.info(f"📈 Training AUC: {round(auc, 4)}")

    version = get_next_version()
    model_path = f"{MODEL_REGISTRY}/{version}"

    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, f"{model_path}/model.pkl")

    # Save metadata
    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "auc": auc
    }

    joblib.dump(metadata, f"{model_path}/metadata.pkl")
    save_version(version)

    logger.info(f"✅ Model saved as {version}")

    return version, auc
