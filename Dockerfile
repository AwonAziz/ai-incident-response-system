FROM python:3.11-slim

LABEL maintainer="Awon Aziz <awonaziz786@gmail.com>"
LABEL description="AI-Powered Incident Response System"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create required directories
RUN mkdir -p logs models data/sample

# Train the model at build time so the container is ready to run
RUN python scripts/train_model.py --samples 300

EXPOSE 8080

# Default: run headless pipeline
CMD ["python", "main.py", "--no-dashboard"]
