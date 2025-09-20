# # Use Python 3.11 slim image
# FROM python:3.11-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# # Install system dependencies for audio processing and general functionality
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     curl \
#     git \
#     ffmpeg \
#     libsndfile1 \
#     libsndfile1-dev \
#     libasound2-dev \
#     portaudio19-dev \
#     libffi-dev \
#     libssl-dev \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy Docker-specific requirements
# COPY requirements.docker.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.docker.txt

# # Copy the application code
# COPY app/ ./app/
# COPY engine/ ./engine/
# COPY ai/ ./ai/
# COPY alembic/ ./alembic/
# COPY alembic.ini ./
# COPY main.py ./
# COPY .env.example .env

# # Create necessary directories
# RUN mkdir -p uploads downloads logs storage data \
#     volumes/milvus volumes/etcd volumes/minio

# # Set proper permissions for upload directories
# RUN chmod 755 uploads downloads logs storage data

# # Expose the port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Run database migrations and start the application
# CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DISPLAY=:99

# Install system dependencies for audio + Chrome
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    gnupg \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    unzip \
    # Chrome & Selenium dependencies
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libxrandr2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome stable
RUN wget -q https://dl.google.com/linux/linux_signing_key.pub -O /usr/share/keyrings/google-linux-signing-key.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-key.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Docker-specific requirements
COPY requirements.docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.docker.txt

# Copy the application code
COPY app/ ./app/
COPY engine/ ./engine/
COPY ai/ ./ai/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY main.py ./
COPY .env.example .env

# Create necessary directories
RUN mkdir -p uploads downloads logs storage data \
    volumes/milvus volumes/etcd volumes/minio

# Set proper permissions for upload directories
RUN chmod 755 uploads downloads logs storage data

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run database migrations and start the application
CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]