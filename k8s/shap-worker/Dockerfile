FROM python:3.10-slim

WORKDIR /app

# ===============================
# System dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Python dependencies
# ===============================
RUN pip install --no-cache-dir \
    confluent-kafka \
    shap \
    numpy \
    pandas \
    scikit-learn \
    lightgbm \
    joblib

# ===============================
# Copy FULL src (for feature engineering)
# ===============================
COPY src /app/src
ENV PYTHONPATH=/app

# ===============================
# Copy model artifacts (CORRECT PATH)
# ===============================
COPY models/fraudguard_lightgbm /app/models/fraudguard_lightgbm

# ===============================
# Copy SHAP worker
# ===============================
COPY k8s/shap-worker/worker.py /app/worker.py

CMD ["python", "worker.py"]
