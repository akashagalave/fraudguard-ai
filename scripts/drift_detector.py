"""
Drift Detector — PSI (Population Stability Index) based.

Since live production data is not available, this script uses the test
dataset as a proxy for incoming production data to simulate drift detection.

In a real production system:
- Training data = reference distribution
- Live streaming predictions = actual distribution
- This script would run on a rolling window of live data (e.g. last 7 days)
- Live data would be stored in S3 and pointed to via LIVE_DATA_PATH env var

PSI Interpretation:
  PSI < 0.10  → No significant drift (stable)
  PSI 0.10-0.20 → Moderate drift (monitor closely)
  PSI > 0.20  → Significant drift (retrain required)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("drift-detector")

DRIFT_THRESHOLD    = float(os.getenv("DRIFT_THRESHOLD",  "0.2"))
MODERATE_THRESHOLD = 0.10
RETRAIN_FLAG_PATH  = "retrain.flag"
BINS               = 10

# Live data path — env var se override 
REF_PATH  = os.getenv("REF_DATA_PATH",  "data/processed/train_features_v2.csv")
LIVE_PATH = os.getenv("LIVE_DATA_PATH", "data/processed/test_features_v2.csv")

REPORT_DIR  = Path("reports/drift")
REPORT_PATH = REPORT_DIR / "psi_report.json"

SKIP_COLS = {"is_fraud", "trans_date_trans_time", "dob", "ts", "unix_time", "Unnamed: 0", "trans_num"}


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = BINS) -> float:

    expected = expected.dropna()
    actual   = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Percentile-based breakpoints — reference distribution se
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0]  -= 1e-6   # edge include karo
    breakpoints[-1] += 1e-6

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts,   _ = np.histogram(actual,   bins=breakpoints)

    expected_perc = expected_counts / (len(expected) + 1e-10)
    actual_perc   = actual_counts   / (len(actual)   + 1e-10)

    # Zero buckets handle 
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-6, actual_perc)

    psi_value = np.sum(
        (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    )
    return float(psi_value)


def calculate_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:

    all_cats = set(expected.dropna().unique()) | set(actual.dropna().unique())

    ref_pct = expected.value_counts(normalize=True)
    act_pct = actual.value_counts(normalize=True)

    psi_value = 0.0
    for cat in all_cats:
        p_ref = float(ref_pct.get(cat, 1e-6))
        p_act = float(act_pct.get(cat, 1e-6))
        p_ref = max(p_ref, 1e-6)
        p_act = max(p_act, 1e-6)
        psi_value += (p_act - p_ref) * np.log(p_act / p_ref)

    return float(psi_value)


def main():
    logger.info(f"Reference data : {REF_PATH}")
    logger.info(f"Live/test data : {LIVE_PATH}")
    logger.info(
        "NOTE: Using test data as proxy for live data. "
        "In production, set LIVE_DATA_PATH env var to real streaming data."
    )

    ref  = pd.read_csv(REF_PATH)
    live = pd.read_csv(LIVE_PATH)

   
    common_cols = [
        c for c in ref.columns
        if c in live.columns
        and c not in SKIP_COLS
        and ref[c].dtype in ["int64", "float64", "int32", "float32"]
    ]

    logger.info(f"Computing PSI for {len(common_cols)} numeric features...")

    drift_scores = {}

    for col in common_cols:
        ref_col  = ref[col]
        live_col = live[col]

        # Low cardinality → categorical PSI
        is_categorical = ref_col.nunique() <= 20

        if is_categorical:
            psi_val = calculate_categorical_psi(ref_col, live_col)
        else:
            psi_val = calculate_psi(ref_col, live_col)

        drift_scores[col] = round(psi_val, 6)

    drifted_cols  = {k: v for k, v in drift_scores.items() if v > DRIFT_THRESHOLD}
    moderate_cols = {k: v for k, v in drift_scores.items() if MODERATE_THRESHOLD < v <= DRIFT_THRESHOLD}
    stable_cols   = {k: v for k, v in drift_scores.items() if v <= MODERATE_THRESHOLD}

    max_psi      = max(drift_scores.values()) if drift_scores else 0.0
    max_psi_feat = max(drift_scores, key=drift_scores.get) if drift_scores else "none"

    logger.info(f"Max PSI        : {max_psi:.4f} (feature: {max_psi_feat})")
    logger.info(f"Drifted (>{DRIFT_THRESHOLD})  : {len(drifted_cols)} features")
    logger.info(f"Moderate       : {len(moderate_cols)} features")
    logger.info(f"Stable         : {len(stable_cols)} features")

    if drifted_cols:
        logger.warning("Drifted features:")
        for feat, score in sorted(drifted_cols.items(), key=lambda x: -x[1]):
            logger.warning(f"  {feat}: PSI = {score:.4f}")

    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "computed_at":       datetime.now(timezone.utc).isoformat(),
        "ref_path":          REF_PATH,
        "live_path":         LIVE_PATH,
        "drift_threshold":   DRIFT_THRESHOLD,
        "max_psi":           max_psi,
        "max_psi_feature":   max_psi_feat,
        "n_features_checked": len(common_cols),
        "n_drifted":         len(drifted_cols),
        "n_moderate":        len(moderate_cols),
        "n_stable":          len(stable_cols),
        "drift_detected":    max_psi > DRIFT_THRESHOLD,
        "drifted_features":  drifted_cols,
        "feature_scores":    drift_scores,
    }
    json.dump(report, open(REPORT_PATH, "w"), indent=2)
    logger.info(f"PSI report saved → {REPORT_PATH}")

    # ── retrain.flag ──────────────────────────────────────────────────────────
    if max_psi > DRIFT_THRESHOLD:
        logger.warning(
            f"DRIFT DETECTED (max PSI = {max_psi:.4f}) "
            f"— writing {RETRAIN_FLAG_PATH}"
        )
        with open(RETRAIN_FLAG_PATH, "w") as f:
            f.write("true")
    else:
        logger.info("No significant drift — retraining not required")
        if os.path.exists(RETRAIN_FLAG_PATH):
            os.remove(RETRAIN_FLAG_PATH)

    return report


if __name__ == "__main__":
    main()
