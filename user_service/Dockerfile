# user_service/Dockerfile
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Variables d'environnement pour éviter les fichiers .pyc et activer unbuffered mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installer les dépendances système nécessaires pour psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copier les requirements pour utiliser le cache Docker
COPY user_service/requirements.txt /app/user_service/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r user_service/requirements.txt

# Copier les modules communs (db_service, config_service)
COPY db_service /app/db_service
COPY config_service /app/config_service

# Copier le code du user_service
COPY user_service /app/user_service

# Copier le .env pour la configuration
COPY .env /app/.env

# Exposer le port 3000 pour le user_service
EXPOSE 3000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:3000/health')" || exit 1

# Commande de démarrage
CMD ["uvicorn", "user_service.main:app", "--host", "0.0.0.0", "--port", "3000"]
