
import json
import logging
import joblib
from pathlib import Path

logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "fraudguard_lightgbm"

_model = None
_feature_columns = None
_categorical_cols = None

def load_model_assets():
    global _model, _feature_columns, _categorical_cols

    if _model is not None:
        return _model, _feature_columns, _categorical_cols

    logger.info(f"Loading model from local path: {MODEL_DIR}")
    _model = joblib.load(MODEL_DIR / "model.pkl")
    _feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
    _categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))
    logger.info("Model + metadata loaded successfully")
    return _model, _feature_columns, _categorical_cols
