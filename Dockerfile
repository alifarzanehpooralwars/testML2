FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (optional, lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Disable MLflow inside the container build step
ENV DISABLE_MLFLOW=1

COPY . .

# Generate embeddings at build time
RUN python -m pipeline.embed

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

