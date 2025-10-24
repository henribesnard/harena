#!/bin/bash

# ============================================
# SCRIPT DE DÉPLOIEMENT HARENA BACKEND
# ============================================
# Ce script déploie les 5 services backend sur AWS EC2
# sans écraser PostgreSQL et Redis existants.
#
# Prérequis :
# - Instance EC2 avec Docker et Docker Compose installés
# - PostgreSQL et Redis déjà déployés
# - Elasticsearch migré vers Bonsai (Heroku)
# - Nginx installé
# ============================================

set -e  # Arrêter en cas d'erreur

echo "============================================"
echo "🚀 DÉPLOIEMENT HARENA BACKEND - AWS EC2"
echo "============================================"
echo ""

# ============================================
# 1. Vérification des prérequis
# ============================================
echo "📋 Étape 1/7 : Vérification des prérequis..."

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Installation..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installé"
else
    echo "✅ Docker déjà installé"
fi

# Vérifier Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Installation..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installé"
else
    echo "✅ Docker Compose déjà installé"
fi

# Vérifier Nginx
if ! command -v nginx &> /dev/null; then
    echo "❌ Nginx n'est pas installé. Installation..."
    sudo yum install -y nginx  # Amazon Linux 2023
    echo "✅ Nginx installé"
else
    echo "✅ Nginx déjà installé"
fi

echo ""

# ============================================
# 2. Vérification des conteneurs existants
# ============================================
echo "📦 Étape 2/7 : Vérification des conteneurs existants..."

if docker ps | grep -q "harena-postgres"; then
    echo "✅ PostgreSQL détecté (harena-postgres) - À CONSERVER"
else
    echo "⚠️  PostgreSQL (harena-postgres) non trouvé. Vérifiez votre configuration."
fi

if docker ps | grep -q "harena-redis"; then
    echo "✅ Redis détecté (harena-redis) - À CONSERVER"
else
    echo "⚠️  Redis (harena-redis) non trouvé. Vérifiez votre configuration."
fi

if docker ps | grep -q "harena-elasticsearch"; then
    echo "⚠️  Elasticsearch local détecté. Arrêt recommandé pour libérer de la RAM..."
    read -p "Voulez-vous arrêter Elasticsearch local ? (o/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        docker stop harena-elasticsearch
        docker rm harena-elasticsearch
        echo "✅ Elasticsearch local arrêté. Utilisez Bonsai (Heroku)."
    fi
fi

echo ""

# ============================================
# 3. Vérification du fichier .env
# ============================================
echo "🔑 Étape 3/7 : Vérification du fichier .env..."

if [ ! -f .env ]; then
    echo "❌ Fichier .env manquant. Création depuis le template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT : Configurez le fichier .env avant de continuer !"
    exit 1
else
    echo "✅ Fichier .env trouvé"

    # Vérifier les variables critiques
    if ! grep -q "DATABASE_URL=" .env; then
        echo "⚠️  DATABASE_URL manquant dans .env"
    fi

    if ! grep -q "REDIS_URL=" .env; then
        echo "⚠️  REDIS_URL manquant dans .env"
    fi

    if ! grep -q "ELASTICSEARCH_URL=" .env; then
        echo "⚠️  ELASTICSEARCH_URL manquant dans .env (utilisez Bonsai)"
    fi
fi

echo ""

# ============================================
# 4. Configuration Nginx
# ============================================
echo "🌐 Étape 4/7 : Configuration Nginx..."

if [ -f nginx.conf ]; then
    sudo cp nginx.conf /etc/nginx/conf.d/harena.conf

    # Tester la configuration Nginx
    if sudo nginx -t; then
        echo "✅ Configuration Nginx valide"
        sudo systemctl reload nginx
        echo "✅ Nginx rechargé"
    else
        echo "❌ Configuration Nginx invalide. Vérifiez nginx.conf"
        exit 1
    fi
else
    echo "⚠️  Fichier nginx.conf manquant. Utilisation de la configuration par défaut."
fi

echo ""

# ============================================
# 5. Build des images Docker
# ============================================
echo "🔨 Étape 5/7 : Build des images Docker..."

docker-compose -f docker-compose.prod.yml build

echo "✅ Images Docker buildées"
echo ""

# ============================================
# 6. Démarrage des services
# ============================================
echo "▶️  Étape 6/7 : Démarrage des services..."

docker-compose -f docker-compose.prod.yml up -d

echo "✅ Services démarrés"
echo ""

# ============================================
# 7. Vérification du déploiement
# ============================================
echo "🔍 Étape 7/7 : Vérification du déploiement..."

sleep 10  # Attendre le démarrage des services

echo ""
echo "Conteneurs en cours d'exécution :"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "Logs des services (dernières lignes) :"
docker-compose -f docker-compose.prod.yml logs --tail=5

echo ""
echo "============================================"
echo "✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS !"
echo "============================================"
echo ""
echo "📊 Ressources système :"
free -h
echo ""
echo "🔗 Points d'accès :"
echo "  - User Service:      http://63.35.52.216/api/v1/users"
echo "  - Metric Service:    http://63.35.52.216/api/v1/metrics"
echo "  - Budget Service:    http://63.35.52.216/api/v1/budget"
echo "  - Search Service:    http://63.35.52.216/api/v1/search"
echo "  - Conversation V3:   http://63.35.52.216/api/v3"
echo ""
echo "📝 Commandes utiles :"
echo "  - Voir les logs :    docker-compose -f docker-compose.prod.yml logs -f [service]"
echo "  - Redémarrer :       docker-compose -f docker-compose.prod.yml restart"
echo "  - Arrêter :          docker-compose -f docker-compose.prod.yml down"
echo "  - Mettre à jour :    git pull && docker-compose -f docker-compose.prod.yml up -d --build"
echo ""
echo "🎉 Votre backend Harena est maintenant en production !"
