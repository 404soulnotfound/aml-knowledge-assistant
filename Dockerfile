# Use official lightweight Python image
FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY data/ ./data/
COPY src/ ./src/
COPY .env.example .env

# Create non-root user for security compliance
RUN useradd -u 1001 amluser && \
    mkdir -p /app/chroma_db && \
    chown -R amluser:amluser /app

USER amluser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/metrics || exit 1

# Start FastAPI application with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
