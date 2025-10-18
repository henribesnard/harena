# sync_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY sync_service/requirements.txt /app/sync_service/requirements.txt
RUN pip install --no-cache-dir -r sync_service/requirements.txt

COPY db_service /app/db_service
COPY config_service /app/config_service
COPY sync_service /app/sync_service
COPY .env /app/.env

EXPOSE 3004

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3004/health || exit 1

CMD ["uvicorn", "sync_service.main:app", "--host", "0.0.0.0", "--port", "3004"]
