
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



DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
MODERATE_THRESHOLD  = 0.10
RETRAIN_FLAG_PATH   = "retrain.flag"
DRIFT_REPORT_DIR    = Path("reports/drift")
DRIFT_REPORT_PATH   = DRIFT_REPORT_DIR / "psi_report.json"
BINS                = 10

REF_PATH  = os.getenv("REF_DATA_PATH",  "data/processed/train_features_v2.csv")
LIVE_PATH = os.getenv("LIVE_DATA_PATH", "data/processed/live.csv")

SKIP_COLS = {"is_fraud", "trans_date_trans_time", "dob", "ts", "unix_time", "Unnamed: 0"}

def compute_psi(reference: pd.Series, actual: pd.Series, bins: int = BINS) -> float:
    """
    Compute PSI between reference distribution and actual distribution.

    Args:
        reference : values from training / reference dataset
        actual    : values from live / production dataset
        bins      : number of histogram buckets

    Returns:
        PSI score (float). Higher = more drift.
    """

    reference = reference.dropna()
    actual = actual.dropna()

    if len(reference) == 0 or len(actual) == 0:
        return 0.0


    min_val = min(reference.min(), actual.min())
    max_val = max(reference.max(), actual.max())

    if min_val == max_val:
    
        return 0.0

    breakpoints = np.linspace(min_val, max_val, bins + 1)

  
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    act_counts, _ = np.histogram(actual,    bins=breakpoints)


    ref_pct = ref_counts / (len(reference) + 1e-10)
    act_pct = act_counts / (len(actual)    + 1e-10)


    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi_value = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))

    return float(psi_value)



def compute_categorical_psi(reference: pd.Series, actual: pd.Series) -> float:
    """
    PSI for categorical / low-cardinality features.
    Uses value frequency instead of histogram bins.
    """
    all_cats = set(reference.dropna().unique()) | set(actual.dropna().unique())

    ref_pct = reference.value_counts(normalize=True)
    act_pct = actual.value_counts(normalize=True)

    psi_value = 0.0
    for cat in all_cats:
        p_ref = float(ref_pct.get(cat, 1e-6))
        p_act = float(act_pct.get(cat, 1e-6))
        if p_ref < 1e-8:
            p_ref = 1e-6
        if p_act < 1e-8:
            p_act = 1e-6
        psi_value += (p_act - p_ref) * np.log(p_act / p_ref)

    return float(psi_value)



def main():
    logger.info(f"Loading reference data from: {REF_PATH}")
    logger.info(f"Loading live data from:      {LIVE_PATH}")

    ref  = pd.read_csv(REF_PATH)
    live = pd.read_csv(LIVE_PATH)

    # Only compare common columns (excluding skip list)
    common_cols = [
        c for c in ref.columns
        if c in live.columns and c not in SKIP_COLS
    ]

    logger.info(f"Computing PSI for {len(common_cols)} features...")

    drift_scores = {}

    for col in common_cols:
        ref_col  = ref[col]
        live_col = live[col]

        dtype = ref_col.dtype
        is_categorical = (
            dtype == "object"
            or str(dtype) == "category"
            or ref_col.nunique() <= 20
        )

        if is_categorical:
            psi_val = compute_categorical_psi(ref_col, live_col)
        else:
            psi_val = compute_psi(ref_col, live_col)

        drift_scores[col] = round(psi_val, 6)


    drifted_cols   = {k: v for k, v in drift_scores.items() if v > DRIFT_THRESHOLD}
    moderate_cols  = {k: v for k, v in drift_scores.items() if MODERATE_THRESHOLD < v <= DRIFT_THRESHOLD}
    stable_cols    = {k: v for k, v in drift_scores.items() if v <= MODERATE_THRESHOLD}

    max_psi      = max(drift_scores.values()) if drift_scores else 0.0
    max_psi_feat = max(drift_scores, key=drift_scores.get) if drift_scores else "none"

    logger.info(f"Max PSI: {max_psi:.4f} (feature: {max_psi_feat})")
    logger.info(f"Drifted features (PSI > {DRIFT_THRESHOLD}): {len(drifted_cols)}")
    logger.info(f"Moderate features ({MODERATE_THRESHOLD}–{DRIFT_THRESHOLD}): {len(moderate_cols)}")
    logger.info(f"Stable features (< {MODERATE_THRESHOLD}): {len(stable_cols)}")

    if drifted_cols:
        logger.warning("Drifted features:")
        for feat, score in sorted(drifted_cols.items(), key=lambda x: -x[1]):
            logger.warning(f"  {feat}: PSI={score:.4f}")


    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "max_psi": max_psi,
        "max_psi_feature": max_psi_feat,
        "drift_threshold": DRIFT_THRESHOLD,
        "n_features_checked": len(common_cols),
        "n_drifted": len(drifted_cols),
        "n_moderate": len(moderate_cols),
        "n_stable": len(stable_cols),
        "drift_detected": max_psi > DRIFT_THRESHOLD,
        "feature_scores": drift_scores,
        "drifted_features": drifted_cols,
    }
    json.dump(report, open(DRIFT_REPORT_PATH, "w"), indent=2)
    logger.info(f"PSI report saved → {DRIFT_REPORT_PATH}")

    if max_psi > DRIFT_THRESHOLD:
        logger.warning(f"DRIFT DETECTED (max PSI={max_psi:.4f}) — writing {RETRAIN_FLAG_PATH}")
        with open(RETRAIN_FLAG_PATH, "w") as f:
            f.write("true")
    else:
        logger.info("No significant drift — no retraining required")
        if os.path.exists(RETRAIN_FLAG_PATH):
            os.remove(RETRAIN_FLAG_PATH)

    return report


if __name__ == "__main__":
    main()
