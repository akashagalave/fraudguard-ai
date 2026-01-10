FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    seldon-core \
    lightgbm \
    joblib \
    numpy \
    shap \
    scikit-learn

COPY models/fraudguard_lightgbm/model.pkl /app/model.pkl
COPY models/fraudguard_lightgbm/feature_columns.json /app/feature_columns.json
COPY k8s/seldon/model.py /app/Model.py

ENV MODEL_NAME=Model
ENV SERVICE_TYPE=MODEL

CMD exec seldon-core-microservice Model --service-type MODEL
