import json
import joblib
import mlflow
from pathlib import Path
from .config import MODEL_NAME, MODEL_STAGE, MLFLOW_TRACKING_URI

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "fraudguard_lightgbm"

def load_artifacts():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
    categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))

    return model, feature_columns, categorical_cols
