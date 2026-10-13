# ==============================================================================
# MORUS-LENS: Mulberry Leaf Disease Intelligence & Agronomic Portal
# Production Dockerfile
# ==============================================================================

# 1. Base Image: Use official lightweight Python 3.11 slim
FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production

# 2. Install essential system dependencies required for OpenCV headless and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory inside container
WORKDIR /app

# 4. Copy and install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . /app/

# 6. Ensure runtime directories exist with appropriate permissions
RUN mkdir -p /app/instance /app/static/uploads

# 7. Expose application port
EXPOSE 5000

# 8. Healthcheck to monitor container availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# 9. Production execution with Gunicorn WSGI server (dynamic PORT for Render, single worker + threads for 512MB RAM safety)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app"]
