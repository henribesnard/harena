FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système pour PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements depuis conversation_service_v3
COPY conversation_service_v3/requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier les dépendances partagées (db_service, config_service, enrichment_service)
COPY db_service /app/db_service
COPY config_service /app/config_service
COPY enrichment_service /app/enrichment_service

# Copier le code de l'application v3
COPY conversation_service_v3/app /app/app

# Exposer le port
EXPOSE 3008

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3008"]
