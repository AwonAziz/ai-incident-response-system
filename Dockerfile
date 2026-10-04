FROM python:3.11-slim AS base

LABEL maintainer="Awon Aziz <awonaziz786@gmail.com>"
LABEL description="AI-Powered Incident Response System"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer is cached across source changes. Every
# requirement ships a manylinux wheel, so no compiler toolchain is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user; the model and logs are written to mounted volumes.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs /app/models \
    && chown -R appuser:appuser /app
USER appuser

# Bake the model into the image so the container starts ready to serve.
# Override with a bind mount (- ./models:/app/models) to use a local model.
RUN python scripts/train_model.py --samples 300

ENV API_ENABLED=true \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    LOG_LEVEL=INFO \
    COLLECTION_INTERVAL=5

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"

CMD ["python", "main.py", "--no-dashboard", "--api"]