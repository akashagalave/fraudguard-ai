import pandas as pd
import numpy as np
import logging
from pathlib import Path


logger = logging.getLogger("fraud_preprocessing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

PII_COLS = ["first", "last", "street", "city", "state", "zip"]

def load_data(path: Path):
    df = pd.read_csv(
        path, 
        engine="python", 
        on_bad_lines="skip",
        encoding_errors="ignore"
    )
    logger.info(f"Loaded {path} | Shape = {df.shape}")
    return df


def drop_pii(df):
    cols = [c for c in PII_COLS if c in df.columns]
    df = df.drop(columns=cols)
    logger.info(f"Removed PII columns: {cols}")
    return df


def process_dates(df):
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["trans_date_trans_time", "dob"])
    logger.info(f"Dropped {before - len(df)} rows due to invalid dates")

    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365).clip(0, 120)


    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype("int8")

    logger.info("Created date/time/age features")
    return df


def clean_geo(df):
    before = len(df)

    mask = (
        df["lat"].between(-90, 90)
        & df["long"].between(-180, 180)
        & df["merch_lat"].between(-90, 90)
        & df["merch_long"].between(-180, 180)
    )

    df = df.loc[mask]
    logger.info(f"Removed {before - len(df)} rows with invalid geo coordinates")
    return df


def fill_missing_fraud(df):
    fraud_sensitive_cols = ["amt", "lat", "long", "merch_lat", "merch_long"]
    
    for col in fraud_sensitive_cols:
        df[f"{col}_missing"] = df[col].isna().astype("int8")

    df["amt"] = df["amt"].fillna(0)  

    df = df.sort_values(["cc_num", "trans_date_trans_time"])
    geo_cols = ["lat", "long", "merch_lat", "merch_long"]

    for col in geo_cols:
        df[col] = df.groupby("cc_num")[col].ffill()

    for col in geo_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(f"unknown_{col}")

    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].fillna(0)

    logger.info("Applied fraud-aware missing handling")
    return df


def save(path, df):
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned file: {path}")


if __name__ == "__main__":

    ROOT = Path(__file__).parent.parent.parent

    raw_train = ROOT / "data" / "raw" / "train.csv"
    raw_test  = ROOT / "data" / "raw" / "test.csv"

    interim = ROOT / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    df = load_data(raw_train)
    df = drop_pii(df)
    df = process_dates(df)
    df = clean_geo(df)
    df = fill_missing_fraud(df)
    df = df.drop_duplicates()

    save(interim / "train_cleaned.csv", df)

    df = load_data(raw_test)
    df = drop_pii(df)
    df = process_dates(df)
    df = clean_geo(df)
    df = fill_missing_fraud(df)
    df = df.drop_duplicates()

    save(interim / "test_cleaned.csv", df)

    logger.info(" DATA PREPROCESSING COMPLETED SUCCESSFULLY")
