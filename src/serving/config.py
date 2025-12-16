MLFLOW_TRACKING_URI = "https://dagshub.com/akashagalaveaaa1/fraudguard-ai.mlflow"

MODEL_NAME = "fraudguard_lightgbm"
MODEL_STAGE = "Production"

FRAUD_THRESHOLD = 0.7

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_TTL_SECONDS = 3600  

FEATURE_COLUMNS_PATH = "models/fraudguard_lightgbm/feature_columns.json"
CATEGORICAL_COLS_PATH = "models/fraudguard_lightgbm/categorical_cols.json"
