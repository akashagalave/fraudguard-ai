import json
import mlflow
from pathlib import Path
from typing import Tuple, List

from .config import (
    MODEL_NAME,
    MODEL_STAGE,
    MLFLOW_TRACKING_URI,
    FEATURE_COLUMNS_PATH,
    CATEGORICAL_COLS_PATH,
)

BASE_DIR = Path(__file__).resolve().parents[2]


def load_artifacts():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    feature_columns = json.load(open(BASE_DIR / FEATURE_COLUMNS_PATH))
    categorical_cols = json.load(open(BASE_DIR / CATEGORICAL_COLS_PATH))

    return model, feature_columns, categorical_cols
