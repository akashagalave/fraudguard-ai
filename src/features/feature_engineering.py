import pandas as pd
import numpy as np
import logging
import mlflow

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Starting Tier-1 Feature Engineering")

    df["distance_km"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    ) * 111

    df["amt_log"] = np.log1p(df["amt"])
    df["amt_zscore"] = (df["amt"] - df["amt"].mean()) / df["amt"].std()

    df["amt_bucket"] = pd.qcut(
        df["amt"], q=5, labels=[0,1,2,3,4]
    ).astype(int)

    df["hour_bucket"] = pd.cut(
        df["hour"],
        bins=[-1,5,11,17,23],
        labels=["night","morning","afternoon","evening"]
    )

    df["day_of_week_bucket"] = df["day_of_week"].map({
        0:"Mon-Tue",1:"Mon-Tue",
        2:"Wed-Thu",3:"Wed-Thu",
        4:"Fri",5:"Weekend",6:"Weekend"
    })

    merchant_stats = df.groupby("merchant")["is_fraud"].agg(["mean","count"])
    df["merchant_fraud_rate"] = df["merchant"].map(merchant_stats["mean"])
    df["merchant_txn_count"] = df["merchant"].map(merchant_stats["count"])

    df["city_pop_bucket"] = pd.qcut(df["city_pop"], q=10, labels=False)

    logger.info("Tier-1 completed")
    return df


def tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Starting Tier-2 Feature Engineering")

    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)

    df["time_since_prev"] = (
        df.groupby("cc_num")["ts"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    def rolling_count(hours):
        return (
            df.groupby("cc_num")
              .rolling(f"{hours}h", on="ts")["amt"]
              .count()
              .reset_index(level=0, drop=True)
              .values
        )

    def rolling_sum(hours):
        return (
            df.groupby("cc_num")
              .rolling(f"{hours}h", on="ts")["amt"]
              .sum()
              .reset_index(level=0, drop=True)
              .values
        )

    df["txn_1h"]  = rolling_count(1)
    df["txn_6h"]  = rolling_count(6)
    df["txn_24h"] = rolling_count(24)

    df["amt_1h"]  = rolling_sum(1)
    df["amt_6h"]  = rolling_sum(6)
    df["amt_24h"] = rolling_sum(24)

    df["amt_mean_24h"] = df["amt_24h"] / df["txn_24h"].replace(0, 1)

    df["txn_burst_flag"] = (df["txn_1h"] >= 5).astype(int)
    df["amt_to_mean_ratio"] = df["amt"] / df["amt_mean_24h"].replace(0, 1)
    df["txn_ratio_1h_24h"] = df["txn_1h"] / df["txn_24h"].replace(0, 1)

    logger.info("Tier-2 completed")
    return df


def tier3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Starting Tier-3 Feature Engineering")

    amt_stats = df.groupby("cc_num")["amt"].agg(
        card_amt_mean="mean",
        card_amt_std="std"
    )
    df = df.merge(amt_stats, on="cc_num", how="left")
    df["card_amt_std"] = df["card_amt_std"].fillna(1)

    df["card_amt_zscore"] = (
        df["amt"] - df["card_amt_mean"]
    ) / df["card_amt_std"]

    dist_stats = df.groupby("cc_num")["distance_km"].agg(
        card_dist_mean="mean",
        card_dist_std="std"
    )
    df = df.merge(dist_stats, on="cc_num", how="left")
    df["card_dist_std"] = df["card_dist_std"].fillna(1)

    df["distance_anomaly"] = (
        df["distance_km"] - df["card_dist_mean"]
    ) / df["card_dist_std"]

    time_stats = df.groupby("cc_num").agg(
        night_ratio=("is_night", "mean"),
        weekend_ratio=("is_weekend", "mean")
    )
    df = df.merge(time_stats, on="cc_num", how="left")

    logger.info("Tier-3 completed")
    return df


def build_features(
    df: pd.DataFrame,
    log_to_mlflow: bool = False,
    feature_version: str = "v1"
) -> pd.DataFrame:
    
    logger.info("Starting full feature engineering pipeline")

    df = tier1_features(df)
    df = tier2_features(df)
    df = tier3_features(df)

    if log_to_mlflow and mlflow.active_run():
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_param("num_rows", df.shape[0])
        mlflow.log_param("num_features", df.shape[1])

    logger.info("Feature engineering pipeline completed")
    return df
