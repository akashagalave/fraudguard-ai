import pandas as pd
import numpy as np
import logging
import json
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger("model_building")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

DATA_PATH = "data/processed/train_features_v2.csv"
MODEL_DIR = Path("models/fraudguard_lightgbm")
SHAP_DIR = Path("reports/shap")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

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

categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

params = {
    "n_estimators": 414,
    "learning_rate": 0.0781,
    "max_depth": 10,
    "num_leaves": 84,
    "min_child_samples": 42,
    "subsample": 0.97,
    "colsample_bytree": 0.62,
    "reg_alpha": 5e-4,
    "reg_lambda": 0.024,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

model = LGBMClassifier(**params)
model.fit(X_train, y_train, categorical_feature=categorical_cols)

y_prob = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

logger.info(f"ROC-AUC: {roc_auc:.4f}")
logger.info(f"PR-AUC : {pr_auc:.4f}")

joblib.dump(model, MODEL_DIR / "model.pkl")
json.dump(list(X.columns), open(MODEL_DIR / "feature_columns.json", "w"))
json.dump(categorical_cols, open(MODEL_DIR / "categorical_cols.json", "w"))


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(SHAP_DIR / "shap_global_importance.csv", index=False)

shap.summary_plot(shap_values, X_val, show=False)
plt.savefig(SHAP_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.savefig(SHAP_DIR / "shap_summary_bar.png", bbox_inches="tight")
plt.close()

logger.info(" Model training completed safely")
