FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    confluent-kafka \
    prometheus-client

COPY k8s/alert-worker/worker.py /app/worker.py

CMD ["python", "worker.py"]
