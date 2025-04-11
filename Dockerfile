FROM python:3.13

WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Exposer le port pour l'API
EXPOSE 8001

# Commande pour exécuter les migrations puis lancer l'API
CMD alembic upgrade head && uvicorn user_service.main:app --host 0.0.0.0 --port 8001