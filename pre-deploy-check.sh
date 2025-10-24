#!/bin/bash

# ============================================
# SCRIPT DE VÉRIFICATION PRÉ-DÉPLOIEMENT
# ============================================
# Ce script vérifie que l'environnement est prêt
# pour le déploiement des services backend Harena
# ============================================

set -e

echo "============================================"
echo "🔍 VÉRIFICATION PRÉ-DÉPLOIEMENT HARENA"
echo "============================================"
echo ""

EXIT_CODE=0

# ============================================
# 1. Vérifier les conteneurs existants
# ============================================
echo "📦 1. Vérification des conteneurs existants..."

if docker ps | grep -q "harena-postgres"; then
    echo "  ✅ PostgreSQL (harena-postgres) - Running"
else
    echo "  ❌ PostgreSQL (harena-postgres) - NOT FOUND ou STOPPED"
    echo "     Action : Démarrer PostgreSQL avant de continuer"
    EXIT_CODE=1
fi

if docker ps | grep -q "harena-redis"; then
    echo "  ✅ Redis (harena-redis) - Running"
else
    echo "  ❌ Redis (harena-redis) - NOT FOUND ou STOPPED"
    echo "     Action : Démarrer Redis avant de continuer"
    EXIT_CODE=1
fi

if docker ps | grep -q "harena-elasticsearch"; then
    echo "  ⚠️  Elasticsearch local détecté - Consomme 1.14GB RAM"
    echo "     Recommandation : Arrêter et utiliser Bonsai (Heroku)"
    echo "     Commande : docker stop harena-elasticsearch && docker rm harena-elasticsearch"
fi

echo ""

# ============================================
# 2. Vérifier la RAM disponible
# ============================================
echo "💾 2. Vérification de la RAM disponible..."

AVAILABLE_RAM_MB=$(free -m | awk 'NR==2{print $7}')
REQUIRED_RAM_MB=1200

echo "  RAM disponible : ${AVAILABLE_RAM_MB}MB"
echo "  RAM requise : ${REQUIRED_RAM_MB}MB"

if [ "$AVAILABLE_RAM_MB" -lt "$REQUIRED_RAM_MB" ]; then
    echo "  ❌ RAM insuffisante !"
    echo "     Solutions :"
    echo "     1. Arrêter Elasticsearch local (libère 1.14GB)"
    echo "     2. Upgrader l'instance à t4g.medium (4GB RAM)"
    EXIT_CODE=1
else
    echo "  ✅ RAM suffisante"
fi

echo ""

# ============================================
# 3. Vérifier le disque disponible
# ============================================
echo "💿 3. Vérification du disque disponible..."

AVAILABLE_DISK_GB=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
REQUIRED_DISK_GB=5

echo "  Disque disponible : ${AVAILABLE_DISK_GB}GB"
echo "  Disque requis : ${REQUIRED_DISK_GB}GB"

if [ "$AVAILABLE_DISK_GB" -lt "$REQUIRED_DISK_GB" ]; then
    echo "  ❌ Disque insuffisant !"
    echo "     Action : Nettoyer les images Docker inutilisées"
    echo "     Commande : docker system prune -a"
    EXIT_CODE=1
else
    echo "  ✅ Disque suffisant"
fi

echo ""

# ============================================
# 4. Vérifier le fichier .env
# ============================================
echo "🔑 4. Vérification du fichier .env..."

if [ ! -f .env ]; then
    echo "  ❌ Fichier .env manquant !"
    echo "     Action : Créer le fichier .env avec les variables nécessaires"
    EXIT_CODE=1
else
    echo "  ✅ Fichier .env trouvé"

    # Vérifier les variables critiques
    MISSING_VARS=()

    if ! grep -q "DATABASE_URL=" .env; then
        MISSING_VARS+=("DATABASE_URL")
    fi

    if ! grep -q "REDIS_URL=" .env; then
        MISSING_VARS+=("REDIS_URL")
    fi

    if ! grep -q "ELASTICSEARCH_URL=" .env; then
        MISSING_VARS+=("ELASTICSEARCH_URL")
    fi

    if ! grep -q "SECRET_KEY=" .env; then
        MISSING_VARS+=("SECRET_KEY")
    fi

    if ! grep -q "OPENAI_API_KEY=" .env; then
        MISSING_VARS+=("OPENAI_API_KEY")
    fi

    if ! grep -q "DEEPSEEK_API_KEY=" .env; then
        MISSING_VARS+=("DEEPSEEK_API_KEY")
    fi

    if [ ${#MISSING_VARS[@]} -gt 0 ]; then
        echo "  ⚠️  Variables manquantes dans .env :"
        for var in "${MISSING_VARS[@]}"; do
            echo "     - $var"
        done
        EXIT_CODE=1
    else
        echo "  ✅ Toutes les variables critiques sont présentes"
    fi
fi

echo ""

# ============================================
# 5. Vérifier Docker et Docker Compose
# ============================================
echo "🐳 5. Vérification de Docker..."

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "  ✅ Docker installé : $DOCKER_VERSION"
else
    echo "  ❌ Docker n'est pas installé"
    echo "     Action : Installer Docker"
    echo "     Commande : curl -fsSL https://get.docker.com | sh"
    EXIT_CODE=1
fi

if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo "  ✅ Docker Compose installé : $COMPOSE_VERSION"
else
    echo "  ❌ Docker Compose n'est pas installé"
    echo "     Action : Installer Docker Compose"
    EXIT_CODE=1
fi

echo ""

# ============================================
# 6. Vérifier Nginx
# ============================================
echo "🌐 6. Vérification de Nginx..."

if command -v nginx &> /dev/null; then
    NGINX_VERSION=$(nginx -v 2>&1)
    echo "  ✅ Nginx installé : $NGINX_VERSION"

    # Vérifier si Nginx est en cours d'exécution
    if systemctl is-active --quiet nginx; then
        echo "  ✅ Nginx est en cours d'exécution"
    else
        echo "  ⚠️  Nginx n'est pas en cours d'exécution"
        echo "     Action : Démarrer Nginx"
        echo "     Commande : sudo systemctl start nginx"
    fi
else
    echo "  ❌ Nginx n'est pas installé"
    echo "     Action : Installer Nginx"
    echo "     Commande : sudo yum install -y nginx (Amazon Linux)"
    echo "              : sudo apt install -y nginx (Ubuntu)"
    EXIT_CODE=1
fi

echo ""

# ============================================
# 7. Vérifier la connexion aux bases de données
# ============================================
echo "🔌 7. Vérification de la connexion aux bases de données..."

# Test PostgreSQL
if docker exec harena-postgres pg_isready -U harena_admin &> /dev/null; then
    echo "  ✅ PostgreSQL accessible"
else
    echo "  ❌ PostgreSQL non accessible"
    EXIT_CODE=1
fi

# Test Redis
if docker exec harena-redis redis-cli --no-auth-warning -a "HaReNa2024-Redis-Auth-Token-Secure-Key-123456" ping &> /dev/null; then
    echo "  ✅ Redis accessible"
else
    echo "  ❌ Redis non accessible"
    EXIT_CODE=1
fi

# Test Elasticsearch (Bonsai)
if curl -s -o /dev/null -w "%{http_code}" "https://37r8v9zfzn:4o7ydjkcc8@fir-178893546.eu-west-1.bonsaisearch.net:443" | grep -q "200\|401"; then
    echo "  ✅ Elasticsearch (Bonsai) accessible"
else
    echo "  ⚠️  Elasticsearch (Bonsai) non accessible"
    echo "     Vérifier l'URL dans .env : ELASTICSEARCH_URL"
fi

echo ""

# ============================================
# 8. Vérifier les fichiers nécessaires
# ============================================
echo "📁 8. Vérification des fichiers nécessaires..."

REQUIRED_FILES=(
    "docker-compose.prod.yml"
    "nginx.conf"
    "user_service/Dockerfile"
    "metric_service/Dockerfile"
    "budget_profiling_service/Dockerfile"
    "search_service/Dockerfile"
    "conversation_service_v3/Dockerfile"
)

ALL_FILES_PRESENT=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file manquant"
        ALL_FILES_PRESENT=false
        EXIT_CODE=1
    fi
done

echo ""

# ============================================
# 9. Résumé
# ============================================
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TOUS LES CHECKS SONT PASSÉS !"
    echo "============================================"
    echo ""
    echo "🚀 Vous pouvez procéder au déploiement :"
    echo "   ./deploy.sh"
    echo ""
else
    echo "❌ CERTAINS CHECKS ONT ÉCHOUÉ"
    echo "============================================"
    echo ""
    echo "⚠️  Corrigez les problèmes ci-dessus avant de déployer."
    echo ""
fi

exit $EXIT_CODE
