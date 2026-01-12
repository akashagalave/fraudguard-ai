import json
import logging
from pathlib import Path
from typing import Tuple, List

import mlflow
import dagshub

from .config import MODEL_NAME, MODEL_STAGE

logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "fraudguard_lightgbm"

_model = None
_feature_columns = None
_categorical_cols = None


def load_model_assets():
    global _model, _feature_columns, _categorical_cols

    if _model is not None:
        return _model, _feature_columns, _categorical_cols

    dagshub.init(
        repo_owner="akashagalaveaaa1",
        repo_name="fraudguard-ai",
        mlflow=True
    )

    logger.info("Loading model from MLflow Registry")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    _model = mlflow.sklearn.load_model(model_uri)

    _feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
    _categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))

    logger.info("Model + metadata loaded successfully")
    return _model, _feature_columns, _categorical_cols
