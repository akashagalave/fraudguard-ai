FROM python:3.10-slim

WORKDIR /app

COPY services/producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/producer/producer.py .

CMD ["python", "producer.py"]
