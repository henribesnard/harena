FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY ./app ./app

# Exposer le port
EXPOSE 3008

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3008"]
