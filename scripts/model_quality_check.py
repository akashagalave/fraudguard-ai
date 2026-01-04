# scripts/model_quality_check.py
import os
import json
METRICS_FILE = "reports/evaluation/metrics.json"
if not os.path.exists(METRICS_FILE):
    METRICS_FILE = "metrics.json"  # fallback for local

with open(METRICS_FILE) as f:
    metrics = json.load(f)


roc_auc = metrics.get("roc_auc", 0)
pr_auc = metrics.get("pr_auc", 0)

if roc_auc < 0.9 or pr_auc < 0.9:
    print(f"Model failed quality check: ROC-AUC={roc_auc}, PR-AUC={pr_auc}")
    exit(1)
else:
    print(f"Model passed quality check: ROC-AUC={roc_auc}, PR-AUC={pr_auc}")
