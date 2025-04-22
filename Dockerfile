FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including FFmpeg and git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy application code
COPY backend/ /app/backend/
COPY static/ /app/static/

# Add note about Mistral model
RUN echo "Note: You must mount the model directory with the Mistral model file when running this container." > /app/IMPORTANT.txt

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=tiny \
    MISTRAL_MODEL_PATH=/app/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    MISTRAL_N_CTX=4096 \
    URA_HOST=0.0.0.0 \
    URA_PORT=8080

# Make backend scripts executable
RUN chmod +x /app/backend/*.py

# Create a symbolic link to the static directory
RUN ln -s /app/static /app/backend/static

# Set working directory to backend
WORKDIR /app/backend

# Expose the service port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["python", "run_ura_service.py"] 