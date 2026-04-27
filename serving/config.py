
from dotenv import load_dotenv
load_dotenv()
import os
REDIS_HOST = os.getenv("APP_REDIS_HOST", "localhost")
_raw_port = os.getenv("APP_REDIS_PORT", "6379")
try:
    REDIS_PORT = int(_raw_port)
except ValueError:
    
    REDIS_PORT = int(_raw_port.split(":")[-1])

REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 3600))


KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"

MLFLOW_TRACKING_URI = "https://dagshub.com/akashagalaveaaa1/fraudguard-ai.mlflow"

FEATURE_COLUMNS_PATH = "models/fraudguard_lightgbm/feature_columns.json"
CATEGORICAL_COLS_PATH = "models/fraudguard_lightgbm/categorical_cols.json"

MODEL_NAME = "fraudguard_lightgbm"
MODEL_STAGE = "Production"
MODEL_VERSION = os.getenv("MODEL_VERSION", "fraudguard_v1")

FRAUD_THRESHOLD = 0.3


REDIS_DB = int(os.getenv("REDIS_DB", 0))



EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

ALERT_RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL")

