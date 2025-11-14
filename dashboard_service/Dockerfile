# dashboard_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config_service /app/config_service
COPY dashboard_service /app/dashboard_service
COPY dashboard_service/main.py /app/main.py
COPY .env /app/.env

EXPOSE 3009

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3009/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3009"]

