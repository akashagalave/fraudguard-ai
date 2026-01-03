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
logger.addHandler(handler)

ROOT = Path(__file__).parent.parent.parent
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IN = INTERIM_DIR / "train_cleaned.csv"
TEST_IN  = INTERIM_DIR / "test_cleaned.csv"

TRAIN_OUT = PROCESSED_DIR / "train_features_v2.csv"
TEST_OUT  = PROCESSED_DIR / "test_features_v2.csv"


def run_feature_engineering():

    train_df = pd.read_csv(TRAIN_IN)
    test_df = pd.read_csv(TEST_IN)

    with mlflow.start_run(run_name="feature_engineering_v2_no_leakage"):

        train_features, artifacts = build_features(
            train_df,
            mode="train",
            log_to_mlflow=True
        )

        test_features, _ = build_features(
            test_df,
            mode="test",
            artifacts=artifacts
        )

        train_features.to_csv(TRAIN_OUT, index=False)
        test_features.to_csv(TEST_OUT, index=False)

        mlflow.log_param("train_rows", train_features.shape[0])
        mlflow.log_param("test_rows", test_features.shape[0])

    logger.info("Feature Engineering completed (NO LEAKAGE)")


if __name__ == "__main__":
    run_feature_engineering()
