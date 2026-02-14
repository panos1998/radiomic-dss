# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
    libglib2.0-0 libx11-6 libxcb1 libxext6 libxrender1 libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code (make local paths available to pip)
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
  # install numpy/versioneer first because some packages (e.g. pyradiomics) need them at build time
  && pip install --no-cache-dir numpy versioneer \
  # disable isolated build environment so the build step can use the preinstalled build deps
  && pip install --no-cache-dir --no-build-isolation -r /app/requirements.txt

# Create non-root user and fix permissions
RUN useradd --create-home --shell /bin/bash appuser \
  && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]