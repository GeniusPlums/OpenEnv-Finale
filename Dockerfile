FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY openenv.yaml ./
COPY role_drift_env/ ./role_drift_env/
COPY data/ ./data/
COPY training/ ./training/
COPY tests/ ./tests/
COPY README.md ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "role_drift_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
