import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


PII_COLS = [
    "first", "last", "street", "city", "state",
    "zip", "cc_num", "trans_num"
]


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded: {path} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PII_COLS if c in df.columns]
    df = df.drop(columns=cols)
    logger.info(f"PII Columns removed: {cols}")
    return df



def process_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df = df.dropna(subset=["trans_date_trans_time", "dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 36
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    logger.info("Datetime processed → created age, hour, day_of_week, is_weekend, is_night")
    return df


def clean_geo(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df[
        df["lat"].between(-90, 90)
        & df["long"].between(-180, 180)
        & df["merch_lat"].between(-90, 90)
        & df["merch_long"].between(-180, 180)
    ]
    removed = before - df.shape[0]
    logger.info(f"Invalid geo rows removed: {removed}")
    return df



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()  
    if "is_fraud" in num_cols:
        num_cols.remove("is_fraud") 
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    engineered = ["age", "hour", "day_of_week", "is_weekend", "is_night"]
    for col in engineered:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
  
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].fillna(0)

    logger.info("Handled ALL missing values using intelligent imputation.")
    return df


def save_data(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned file → {path}")



if __name__ == "__main__":

    ROOT = Path(__file__).parent.parent.parent

    raw_train = ROOT / "data" / "raw" / "train.csv"
    raw_test  = ROOT / "data" / "raw" / "test.csv"

    interim_dir = ROOT / "data" / "interim"
    interim_dir.mkdir(exist_ok=True, parents=True)

    out_train = interim_dir / "train_cleaned.csv"
    out_test  = interim_dir / "test_cleaned.csv"

 
    logger.info("Processing TRAIN Dataset")
    train_df = load_data(raw_train)
    train_df = drop_pii(train_df)
    train_df = process_dates(train_df)
    train_df = clean_geo(train_df)
    train_df = handle_missing_values(train_df)
    train_df = train_df.drop_duplicates()
    save_data(train_df, out_train)


    logger.info("Processing TEST Dataset")
    test_df = load_data(raw_test)
    test_df = drop_pii(test_df)
    test_df = process_dates(test_df)
    test_df = clean_geo(test_df)
    test_df = handle_missing_values(test_df)
    test_df = test_df.drop_duplicates()
    save_data(test_df, out_test)

    logger.info("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
