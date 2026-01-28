# Brooks Trading Coach - Production Dockerfile
# Optimized for fast rebuilds with layer caching

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (rarely changes - cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST (largest dep, rarely changes - cached)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy only requirements file first (for better caching)
COPY requirements-prod.txt .

# Install Python dependencies (cached unless requirements change)
RUN pip install -r requirements-prod.txt

# Copy application code LAST (changes frequently, but deps are cached)
COPY app/ ./app/
COPY config.yaml settings.yaml tickers.txt ./

# Create necessary directories
RUN mkdir -p data outputs imports materials

# Expose port (Render sets $PORT dynamically)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Start command - Render will override with $PORT
CMD ["sh", "-c", "uvicorn app.web.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
