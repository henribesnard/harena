#!/bin/bash

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Copier le fichier .env.example vers .env si .env n'existe pas
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Fichier .env créé. Veuillez le modifier avec vos identifiants Bridge API."
fi

# Initialiser la base de données avec Alembic
alembic upgrade head

# Lancer le serveur
uvicorn user_service.main:app --host 0.0.0.0 --port 8001 --reload