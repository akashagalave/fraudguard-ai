# scripts/model_quality_check.py
import os
import json
import sys

# Try to find metrics file in standard locations
METRICS_FILE = "reports/evaluation/metrics.json"
if not os.path.exists(METRICS_FILE):
    METRICS_FILE = "metrics.json"

if not os.path.exists(METRICS_FILE):
    print("Metrics file not found. CI cannot continue.")
    sys.exit(1)

with open(METRICS_FILE) as f:
    metrics = json.load(f)

roc_auc = metrics.get("roc_auc", 0)
pr_auc = metrics.get("pr_auc", 0)

print(f"Model Metrics: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

if roc_auc < 0.9 or pr_auc < 0.9:
    print(f"Model failed quality check: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    sys.exit(1)
else:
    print(f"Model passed quality check: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
