FROM python:3.10-slim

# ===============================
# System dependencies (LightGBM, SHAP)
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# App directory
# ===============================
WORKDIR /app

# ===============================
# Install serving dependencies only
# ===============================
COPY src/serving/requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ===============================
# Copy application code (NO MODELS)
# ===============================
COPY src /app/src
COPY models/ /app/models/


# ===============================
# Python path
# ===============================
ENV PYTHONPATH=/app

# ===============================
# Expose FastAPI port
# ===============================
EXPOSE 8000

# ===============================
# Start FastAPI
# ===============================
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]