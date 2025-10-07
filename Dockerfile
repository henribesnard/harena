# Multi-stage build for optimized Python backend
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim-bookworm

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create app user (non-root)
RUN useradd -m -u 1000 harena && \
    mkdir -p /app && \
    chown -R harena:harena /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=harena:harena . .

# Switch to non-root user
USER harena

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "local_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
