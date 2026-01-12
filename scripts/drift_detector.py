import pandas as pd
import numpy as np
import os

DRIFT_THRESHOLD = 0.2
RETRAIN_FLAG_PATH = "retrain.flag"

def psi(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / (len(expected) + 1e-6)
    actual_perc = actual_perc / (len(actual) + 1e-6)

    return np.sum(
        (expected_perc - actual_perc) *
        np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )

ref = pd.read_csv("data/processed/train.csv")
live = pd.read_csv("data/processed/live.csv")

drift_scores = {}

for col in ref.columns:
    if col in live.columns:
        drift_scores[col] = psi(ref[col], live[col])

max_drift = max(drift_scores.values())

print(f" Max PSI drift: {max_drift}")


if max_drift > DRIFT_THRESHOLD:
    print(" DRIFT DETECTED — retraining required")
    with open(RETRAIN_FLAG_PATH, "w") as f:
        f.write("true")
else:
    print(" No significant drift")
    if os.path.exists(RETRAIN_FLAG_PATH):
        os.remove(RETRAIN_FLAG_PATH)