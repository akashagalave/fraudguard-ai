import pandas as pd
import logging
from pathlib import Path
import mlflow

from src.features.feature_engineering import build_features

logger = logging.getLogger("feature_engineering_runner")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

ROOT = Path(__file__).parent.parent.parent

INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IN = INTERIM_DIR / "train_cleaned.csv"
TEST_IN  = INTERIM_DIR / "test_cleaned.csv"

TRAIN_OUT = PROCESSED_DIR / "train_features_v1.csv"
TEST_OUT  = PROCESSED_DIR / "test_features_v1.csv"


def run_feature_engineering():
    logger.info("Starting Feature Engineering Pipeline")

    logger.info(f"Loading train data from {TRAIN_IN}")
    train_df = pd.read_csv(TRAIN_IN)

    logger.info(f"Loading test data from {TEST_IN}")
    test_df = pd.read_csv(TEST_IN)

    logger.info(f"Train shape before FE: {train_df.shape}")
    logger.info(f"Test shape before FE: {test_df.shape}")

    with mlflow.start_run(run_name="feature_engineering_v1"):
        train_features = build_features(
            train_df,
            log_to_mlflow=True,
            feature_version="v1"
        )

        test_features = build_features(
            test_df,
            log_to_mlflow=False,
            feature_version="v1"
        )

        train_features.to_csv(TRAIN_OUT, index=False)
        test_features.to_csv(TEST_OUT, index=False)

        mlflow.log_param("train_rows", train_features.shape[0])
        mlflow.log_param("train_features", train_features.shape[1])
        mlflow.log_param("test_rows", test_features.shape[0])
        mlflow.log_param("test_features", test_features.shape[1])

    logger.info("Feature Engineering Completed Successfully")
    logger.info(f"Saved → {TRAIN_OUT}")
    logger.info(f"Saved → {TEST_OUT}")


if __name__ == "__main__":
    run_feature_engineering()
