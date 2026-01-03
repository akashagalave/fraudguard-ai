import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def generate_shap_bar_chart(explainer, model, df, feature_columns, top_k=5):
    shap_values = explainer.shap_values(df)

    # LightGBM binary classifier fix
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_vals = shap_values[0]

    shap_pairs = sorted(
        zip(feature_columns, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    features, impacts = zip(*shap_pairs)

    plt.figure(figsize=(6, 4))
    plt.barh(features, impacts)
    plt.xlabel("SHAP Impact")
    plt.title("Top Fraud Drivers")
    plt.gca().invert_yaxis()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150)
    plt.close()

    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    shap_reason_dicts = [
        {"feature": f, "impact": float(i)}
        for f, i in shap_pairs
    ]

    return img_base64, shap_reason_dicts


def generate_human_explanations(shap_reasons, max_lines=3):
    """
    Converts SHAP features into human-readable reasons.
    GUARANTEED to return up to 3 lines.
    """

    mapping = {
        "amt": "Transaction amount is unusually high compared to normal spending",
        "amt_mean_24h": "Recent spending amount is unusually high",
        "card_amt_mean": "Card spending behavior deviates from normal",
        "hour": "Transaction occurred at an unusual time of day",
        "is_night": "Transaction happened during late night hours",
        "merchant_txn_count": "This is a first-time or rare transaction with this merchant",
        "time_since_prev": "Transaction happened very quickly after a previous one",
        "weekend_ratio": "Transaction pattern is unusual for weekends",
        "age": "Customer profile shows higher fraud risk",
        "category": "Transaction category is commonly associated with fraud",
        "job": "Employment profile indicates higher fraud risk"
    }

    explanations = []

    # Sort by strongest SHAP impact
    shap_reasons = sorted(
        shap_reasons,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    for r in shap_reasons:
        feature = r["feature"]

        if feature in mapping:
            text = mapping[feature]
        else:
            # fallback readable feature name
            text = f"Unusual behavior detected in {feature.replace('_', ' ')}"

        if text not in explanations:
            explanations.append(text)

        if len(explanations) == max_lines:
            break

    # Final safety
    if not explanations:
        explanations.append("Unusual transaction behavior detected")

    return explanations
