import pandas as pd
import numpy as np
import mlflow
import dagshub
import shap
import logging
import matplotlib.pyplot as plt
import warnings
import json
import joblib

from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

logger = logging.getLogger("model_building")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="fraudguard-ai",
    mlflow=True
)

mlflow.set_experiment("fraudguard_lightgbm_training")


DATA_PATH = "data/processed/train_features_v1.csv"

SHAP_DIR = Path("reports/shap")
SHAP_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_DIR = Path("models/fraudguard_lightgbm")
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

logger.info(f"Shape: {df.shape}")
logger.info(f"Fraud rate: {df['is_fraud'].mean():.6f}")

TARGET = "is_fraud"

DROP_COLS = [
    "is_fraud",
    "trans_date_trans_time",
    "dob",
    "ts",
    "unix_time",
    "Unnamed: 0"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df[TARGET]

X.columns = (
    X.columns
    .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

logger.info(f"Categorical features: {categorical_cols}")
logger.info(f"Total features: {X.shape[1]}")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

logger.info(f"Train shape: {X_train.shape}")
logger.info(f"Val shape  : {X_val.shape}")

best_params = {
    "n_estimators": 414,
    "learning_rate": 0.07812891022074141,
    "max_depth": 10,
    "num_leaves": 84,
    "min_child_samples": 42,
    "subsample": 0.9726656602712066,
    "colsample_bytree": 0.6222397023485428,
    "reg_alpha": 0.0005036576805115846,
    "reg_lambda": 0.024341998159808864,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}


with mlflow.start_run(run_name="LightGBM_Final_Training"):

    mlflow.log_params(best_params)

    logger.info("Training LightGBM model...")
    model = LGBMClassifier(**best_params)

    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_cols
    )

    y_prob = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    logger.info(f"ROC-AUC: {roc_auc:.6f}")
    logger.info(f"PR-AUC : {pr_auc:.6f}")

    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name="fraudguard_lightgbm"
    )

    logger.info("Saving model locally for FastAPI usage...")

    joblib.dump(model, LOCAL_MODEL_DIR / "model.pkl")

    with open(LOCAL_MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(LOCAL_MODEL_DIR / "categorical_cols.json", "w") as f:
        json.dump(categorical_cols, f, indent=2)

    logger.info(f"Local model saved at: {LOCAL_MODEL_DIR}")

    logger.info("Running SHAP analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_global_df = pd.DataFrame({
        "feature": X_val.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_csv_path = SHAP_DIR / "shap_global_importance.csv"
    shap_global_df.to_csv(shap_csv_path, index=False)
    mlflow.log_artifact(shap_csv_path)

    plt.figure()
    shap.summary_plot(shap_values, X_val, show=False)
    plt.savefig(SHAP_DIR / "shap_summary.png", bbox_inches="tight")
    mlflow.log_artifact(SHAP_DIR / "shap_summary.png")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig(SHAP_DIR / "shap_summary_bar.png", bbox_inches="tight")
    mlflow.log_artifact(SHAP_DIR / "shap_summary_bar.png")
    plt.close()

    logger.info("Model training, local save & SHAP completed successfully.")
