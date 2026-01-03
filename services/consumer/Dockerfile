FROM python:3.10-slim

WORKDIR /app

COPY services/consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/consumer/consumer.py .

CMD ["python", "consumer.py"]
