FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source
COPY src ./src
COPY api ./api
COPY app ./app

# Copy processed vocabulary (required at runtime)
COPY data/processed/vocab.json ./data/processed/vocab.json

# Copy saved experiment results for the comparison dashboard
COPY results ./results

# Model checkpoints are mounted at runtime via docker-compose volumes.
# If they exist at build time they will also be bundled.
COPY artifacts ./artifacts

# Expose API port and Streamlit port
EXPOSE 8000 8501

# Default: start the FastAPI service.
# Override with docker-compose to run Streamlit separately.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

