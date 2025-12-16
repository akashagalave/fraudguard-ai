import json
import mlflow
import dagshub
import shap
from pathlib import Path

from .config import MODEL_NAME, MODEL_STAGE

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "fraudguard_lightgbm"


def load_artifacts():
    dagshub.init(
        repo_owner="akashagalaveaaa1",
        repo_name="fraudguard-ai",
        mlflow=True
    )

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
    categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))

    explainer = shap.TreeExplainer(model)

    return model, explainer, feature_columns, categorical_cols
