import sys
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

MIN_ROC_AUC = 0.91
MIN_PR_AUC = 0.91

model = joblib.load("models/staging/model.pkl")
X_val = joblib.load("data/X_val.pkl")
y_val = joblib.load("data/y_val.pkl")

y_prob = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR  AUC: {pr_auc:.4f}")

if roc_auc < MIN_ROC_AUC or pr_auc < MIN_PR_AUC:
    print("Model failed performance gate")
    sys.exit(1)

print("Model passed performance gate")
