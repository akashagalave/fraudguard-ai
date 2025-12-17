import os
from dotenv import load_dotenv
load_dotenv()



MLFLOW_TRACKING_URI = "https://dagshub.com/akashagalaveaaa1/fraudguard-ai.mlflow"

FEATURE_COLUMNS_PATH = "models/fraudguard_lightgbm/feature_columns.json"
CATEGORICAL_COLS_PATH = "models/fraudguard_lightgbm/categorical_cols.json"

MODEL_NAME = "fraudguard_lightgbm"
MODEL_STAGE = "Production"


FRAUD_THRESHOLD = 0.00000001


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 3600))


EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

ALERT_RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL")

