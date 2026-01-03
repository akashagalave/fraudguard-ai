import json
import logging
import joblib
import pandas as pd
import mlflow
import dagshub
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="fraudguard-ai",
    mlflow=True
)

mlflow.set_experiment("fraudguard_lightgbm_evaluation")

DATA_PATH = "data/processed/test_features_v2.csv"
MODEL_DIR = Path("models/fraudguard_lightgbm")
REPORT_DIR = Path("reports/evaluation")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(MODEL_DIR / "model.pkl")
feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
categorical_cols = json.load(open(MODEL_DIR / "categorical_cols.json"))

df = pd.read_csv(DATA_PATH)

TARGET = "is_fraud"
DROP_COLS = [
    "is_fraud", "trans_date_trans_time", "dob",
    "ts", "unix_time", "Unnamed: 0"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df[TARGET]

X.columns = (
    X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
             .str.replace(r"_+", "_", regex=True)
             .str.strip("_")
)

X = X[feature_columns]
for col in categorical_cols:
    X[col] = X[col].astype("category")

y_prob = model.predict_proba(X)[:, 1]

metrics = {
    "roc_auc": roc_auc_score(y, y_prob),
    "pr_auc": average_precision_score(y, y_prob)
}

json.dump(metrics, open(REPORT_DIR / "metrics.json", "w"), indent=2)

with mlflow.start_run(run_name="LightGBM_Evaluation") as run:

    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X, y_prob)

    logged_model = mlflow.sklearn.log_model(
        model,
        artifact_path="fraud_model",
        signature=signature,
        input_example=X.head(5)
    )

    run_info = {
        "run_id": run.info.run_id,
        "artifact_path": "fraud_model",
        "model_uri": logged_model.model_uri
    }

    with open("run_information.json", "w") as f:
        json.dump(run_info, f, indent=4)

logger.info("Model evaluation completed")
