import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Optional

# ============================
# OPTIONAL MLFLOW (IMPORTANT)
# ============================
try:
    import mlflow
except ImportError:
    mlflow = None

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def tier1_features(
    df: pd.DataFrame,
    merchant_txn_count: Optional[pd.Series] = None
):
    df = df.copy()
    logger.info("Tier-1 FE started")

    df["distance_km"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    ) * 111

    df["amt_log"] = np.log1p(df["amt"])

    std = df["amt"].std()
    df["amt_zscore"] = (df["amt"] - df["amt"].mean()) / (std if std else 1)

    df["amt_bucket"] = pd.cut(
        df["amt"],
        bins=[-np.inf, 10, 50, 200, 1000, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    if merchant_txn_count is None or merchant_txn_count.empty:
        merchant_txn_count = df.groupby("merchant").size()

    df["merchant_txn_count"] = (
        df["merchant"].map(merchant_txn_count).fillna(1)
    )

    df["city_pop_bucket"] = pd.cut(
        df["city_pop"],
        bins=[-1, 500, 2500, 10000, 50000, 200000, 1000000, np.inf],
        labels=False
    ).fillna(0).astype(int)

    return df, merchant_txn_count


def tier2_features(df: pd.DataFrame):
    df = df.copy()
    logger.info("Tier-2 FE started")

    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)

    df["time_since_prev"] = (
        df.groupby("cc_num")["ts"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    def rolling(col, hours, agg):
        return (
            df.groupby("cc_num")
              .rolling(f"{hours}h", on="ts")[col]
              .agg(agg)
              .reset_index(level=0, drop=True)
              .fillna(0)
              .values
        )

    df["txn_1h"] = rolling("amt", 1, "count")
    df["txn_24h"] = rolling("amt", 24, "count")
    df["amt_24h"] = rolling("amt", 24, "sum")

    df["txn_burst_flag"] = (df["txn_1h"] >= 5).astype(int)
    df["amt_mean_24h"] = df["amt_24h"] / df["txn_24h"].replace(0, 1)

    return df


def fit_card_stats(df: pd.DataFrame):
    logger.info("Fitting card-level statistics (TRAIN ONLY)")
    return {
        "card_amt": df.groupby("cc_num")["amt"].agg(["mean", "std"]),
        "card_dist": df.groupby("cc_num")["distance_km"].agg(["mean", "std"])
    }


def apply_card_stats(df: pd.DataFrame, stats: Optional[Dict]):
    df = df.copy()

    if not stats:
        df["card_amt_mean"] = df["amt"]
        df["card_amt_std"] = 1
        df["card_dist_mean"] = df["distance_km"]
        df["card_dist_std"] = 1
    else:
        df["card_amt_mean"] = df["cc_num"].map(stats["card_amt"]["mean"]).fillna(df["amt"])
        df["card_amt_std"] = df["cc_num"].map(stats["card_amt"]["std"]).fillna(1)
        df["card_dist_mean"] = df["cc_num"].map(stats["card_dist"]["mean"]).fillna(df["distance_km"])
        df["card_dist_std"] = df["cc_num"].map(stats["card_dist"]["std"]).fillna(1)

    df["card_amt_zscore"] = (df["amt"] - df["card_amt_mean"]) / df["card_amt_std"]
    df["distance_anomaly"] = (df["distance_km"] - df["card_dist_mean"]) / df["card_dist_std"]

    return df


def build_features(
    df: pd.DataFrame,
    mode: str,
    artifacts: Optional[Dict] = None,
    log_to_mlflow: bool = False,
    feature_version: str = "v2_no_leakage"
):
    logger.info(f"Building features | mode={mode}")
    df = df.copy()

    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if mode == "train":
        df, merchant_stats = tier1_features(df)
        df = tier2_features(df)
        card_stats = fit_card_stats(df)
        df = apply_card_stats(df, card_stats)

        artifacts = {
            "merchant_txn_count": merchant_stats,
            "card_stats": card_stats
        }

        # ✅ REQUIRED LINE (SAVE FEATURE ARTIFACTS)
        joblib.dump(
            artifacts,
            "models/fraudguard_lightgbm/feature_artifacts.pkl"
        )
        logger.info("✅ Feature artifacts saved")

    else:
        merchant_stats = artifacts.get("merchant_txn_count") if artifacts else None
        card_stats = artifacts.get("card_stats") if artifacts else None

        df, _ = tier1_features(df, merchant_stats)
        df = tier2_features(df)
        df = apply_card_stats(df, card_stats)

    df = df.drop(columns=["cc_num", "merchant", "category"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            df[col] = pd.factorize(df[col])[0]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ✅ SAFE MLflow logging
    if log_to_mlflow and mlflow and mlflow.active_run():
        mlflow.log_param("feature_version", feature_version)

    return df, artifacts
