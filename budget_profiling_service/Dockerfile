# budget_profiling_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY db_service /app/db_service
COPY config_service /app/config_service
COPY budget_profiling_service /app/budget_profiling_service
COPY .env /app/.env

EXPOSE 3006

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3006/health || exit 1

CMD ["uvicorn", "budget_profiling_service.main:app", "--host", "0.0.0.0", "--port", "3006"]
