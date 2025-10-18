# conversation_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY conversation_service/requirements.txt /app/conversation_service/requirements.txt
RUN pip install --no-cache-dir -r conversation_service/requirements.txt

COPY db_service /app/db_service
COPY config_service /app/config_service
COPY conversation_service /app/conversation_service
COPY .env /app/.env

EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3003/health || exit 1

CMD ["uvicorn", "conversation_service.main:app", "--host", "0.0.0.0", "--port", "3003"]
