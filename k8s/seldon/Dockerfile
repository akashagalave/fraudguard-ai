FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY k8s/seldon/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + code
COPY k8s/seldon/model.py /app/model.py
COPY models/fraudguard_lightgbm/model.pkl /models/model.pkl

# Expose model port
EXPOSE 9000

# IMPORTANT: service-type MUST be MODEL
CMD ["seldon-core-microservice", "model.FraudGuardModel", "--service-type", "MODEL"]
