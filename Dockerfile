# JARVIS v3.0 Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data blueprints simulations models uploads

# Create non-root user
RUN groupadd -r jarvis && useradd -r -g jarvis jarvis

# Set up backup cron job
COPY scripts/backup.sh /usr/local/bin/backup.sh
RUN chmod +x /usr/local/bin/backup.sh

# Create cron job for backups (every 12 hours)
RUN echo "0 */12 * * * /usr/local/bin/backup.sh" | crontab -

# Change ownership to non-root user
RUN chown -R jarvis:jarvis /app

# Switch to non-root user
USER jarvis

# Initialize database
RUN python -c "from backend.database.config import init_database; init_database()"

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "backend/main.py"]