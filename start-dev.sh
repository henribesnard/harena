#!/bin/bash

# ============================================
# SCRIPT DE DÉMARRAGE DEV
# ============================================
# Démarre tous les services en mode développement
# avec PostgreSQL et Redis locaux
# ============================================

set -e

echo "🚀 Démarrage de l'environnement DEV Harena"
echo "=========================================="
echo ""

# Vérifier que Docker est démarré
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker n'est pas démarré. Veuillez démarrer Docker Desktop."
    exit 1
fi

# Vérifier que le fichier .env.dev existe
if [ ! -f .env.dev ]; then
    echo "❌ Fichier .env.dev manquant"
    echo "Copiez .env.dev.example vers .env.dev et configurez-le"
    exit 1
fi

# Arrêter les conteneurs existants
echo "🛑 Arrêt des conteneurs existants..."
docker-compose -f docker-compose.dev.yml down 2>/dev/null || true

# Créer les répertoires de données si nécessaire
echo "📁 Création des répertoires de données..."
mkdir -p data/postgres-dev data/redis-dev

# Build et démarrage des services
echo "🔨 Build des images Docker..."
docker-compose -f docker-compose.dev.yml build

echo "▶️  Démarrage des services..."
docker-compose -f docker-compose.dev.yml up -d

# Attendre que PostgreSQL soit prêt
echo "⏳ Attente du démarrage de PostgreSQL..."
timeout=60
counter=0
until docker exec postgres-dev pg_isready -U harena_dev > /dev/null 2>&1; do
    sleep 1
    counter=$((counter + 1))
    if [ $counter -gt $timeout ]; then
        echo "❌ Timeout - PostgreSQL n'a pas démarré"
        exit 1
    fi
done

echo "✅ PostgreSQL prêt"

# Appliquer les migrations
echo "🔄 Application des migrations..."
docker-compose -f docker-compose.dev.yml exec -T user_service alembic upgrade head || echo "⚠️  Migrations non appliquées (service peut-être pas encore prêt)"

# Afficher l'état des services
echo ""
echo "📊 État des services:"
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "✅ Environnement DEV démarré avec succès!"
echo ""
echo "🔗 Points d'accès:"
echo "  - Frontend:          http://localhost:5174"
echo "  - User Service:      http://localhost:3000"
echo "  - Search Service:    http://localhost:3001"
echo "  - Metric Service:    http://localhost:3002"
echo "  - Conversation V3:   http://localhost:3008"
echo "  - Budget Service:    http://localhost:3006"
echo "  - PostgreSQL (DBeaver): localhost:5433"
echo "  - Redis:             localhost:6380"
echo "  - Portainer:         http://localhost:9000"
echo ""
echo "📝 Commandes utiles:"
echo "  - Voir les logs:     docker-compose -f docker-compose.dev.yml logs -f"
echo "  - Arrêter:           docker-compose -f docker-compose.dev.yml down"
echo "  - Rebuild:           docker-compose -f docker-compose.dev.yml up -d --build"
echo ""
