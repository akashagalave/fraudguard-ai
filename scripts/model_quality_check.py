import json
import sys

with open("metrics.json") as f:
    metrics = json.load(f)

roc_auc = metrics["roc_auc"]
pr_auc = metrics["pr_auc"]

print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")

if roc_auc < 0.90 or pr_auc < 0.90:
    print("Model quality check failed")
    sys.exit(1)

print("Model quality check passed")
