MODEL_NAME = "fraudguard_lightgbm"
MODEL_STAGE = "Production"

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/akashagalaveaaa1/fraudguard-ai.mlflow"
)

FEATURE_COLUMNS_PATH = "models/fraudguard_lightgbm/feature_columns.json"
CATEGORICAL_COLS_PATH = "models/fraudguard_lightgbm/categorical_cols.json"

FRAUD_THRESHOLD = 0.7
