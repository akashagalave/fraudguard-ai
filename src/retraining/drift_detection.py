import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift-detection")

# ============================
# Config
# ============================
DRIFT_THRESHOLD = 0.05  # p-value threshold
MIN_SAMPLES = 100

# ============================
# Drift Detection
# ============================
def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Uses Kolmogorov–Smirnov test for numerical drift detection
    """

    if len(production_data) < MIN_SAMPLES:
        logger.warning("⚠️ Not enough production samples for drift detection")
        return False, {}

    drifted_features = {}

    for feature in feature_columns:
        if feature not in production_data.columns:
            continue

        ref_vals = reference_data[feature].dropna()
        prod_vals = production_data[feature].dropna()

        if len(ref_vals) < MIN_SAMPLES or len(prod_vals) < MIN_SAMPLES:
            continue

        stat, p_value = ks_2samp(ref_vals, prod_vals)

        if p_value < DRIFT_THRESHOLD:
            drifted_features[feature] = {
                "p_value": round(p_value, 6),
                "statistic": round(stat, 4),
            }

    drift_detected = len(drifted_features) > 0

    if drift_detected:
        logger.warning("🚨 DATA DRIFT DETECTED")
    else:
        logger.info("✅ No significant drift detected")

    return drift_detected, drifted_features
