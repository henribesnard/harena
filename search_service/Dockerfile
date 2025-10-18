# search_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installer les dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY search_service/requirements.txt /app/search_service/requirements.txt

# Installer dépendances Python
RUN pip install --no-cache-dir -r search_service/requirements.txt

# Copier modules communs
COPY db_service /app/db_service
COPY config_service /app/config_service
COPY user_service /app/user_service

# Copier le code du search_service
COPY search_service /app/search_service

# Copier .env
COPY .env /app/.env

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/v1/search/health || exit 1

CMD ["uvicorn", "search_service.main:app", "--host", "0.0.0.0", "--port", "3001"]
