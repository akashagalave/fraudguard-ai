FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    confluent-kafka \
    shap \
    numpy \
    pandas \
    scikit-learn \
    lightgbm \
    joblib

COPY src /app/src
ENV PYTHONPATH=/app


COPY models/fraudguard_lightgbm /app/models/fraudguard_lightgbm


COPY k8s/shap-worker/worker.py /app/worker.py

CMD ["python", "worker.py"]
