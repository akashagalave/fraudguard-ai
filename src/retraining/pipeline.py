import json
import logging
import pandas as pd

from src.retraining.drift_detection import check_drift
from src.retraining.retrain_model import retrain_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retraining-pipeline")

# ============================
# Config
# ============================
FEATURE_COLUMNS_PATH = "models/fraudguard_lightgbm/feature_cols.json"
TRAINING_DATA_PATH = "data/processed/training_data.csv"
DRIFT_THRESHOLD = 0.2


# ============================
# Load helpers
# ============================
def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH) as f:
        return json.load(f)


def load_training_data():
    df = pd.read_csv(TRAINING_DATA_PATH)
    return df


# ============================
# Pipeline
# ============================
def run_retraining_pipeline():
    logger.info("🚀 Retraining pipeline started")

    df = load_training_data()
    feature_cols = load_feature_columns()

    # ----------------------------
    # Drift Detection
    # ----------------------------
    logger.info("🔍 Checking data drift")

    drift_report = check_drift(
        reference_df=df.sample(frac=0.5, random_state=42),
        current_df=df.sample(frac=0.5, random_state=7),
        threshold=DRIFT_THRESHOLD
    )

    if not drift_report["drift_detected"]:
        logger.info("✅ No significant drift detected. Skipping retraining.")
        return {
            "status": "skipped",
            "reason": "no_drift"
        }

    logger.warning(
        f"⚠️ Drift detected in {len(drift_report['drifted_features'])} features"
    )

    # ----------------------------
    # Retraining
    # ----------------------------
    version, auc = retrain_model(
        train_df=df,
        feature_cols=feature_cols,
        target_col="is_fraud"
    )

    logger.info(f"🎯 Retraining complete. New model: {version}")

    return {
        "status": "retrained",
        "version": version,
        "auc": auc,
        "drifted_features": drift_report["drifted_features"]
    }


# ============================
# CLI entry
# ============================
if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(result)
