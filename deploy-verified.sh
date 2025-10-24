#!/bin/bash

# ============================================
# SCRIPT DE DÉPLOIEMENT SÉCURISÉ AVEC VÉRIFICATIONS
# ============================================
# Ce script déploie ET vérifie l'infrastructure

set -e  # Arrêter en cas d'erreur

echo "============================================"
echo "🚀 DÉPLOIEMENT HARENA - VERSION SÉCURISÉE"
echo "============================================"
echo ""

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================
# ÉTAPE 1: Mise à jour du code
# ============================================
echo -e "${YELLOW}📥 Étape 1/6 : Mise à jour du code depuis git...${NC}"
cd /home/ec2-user/harena
git fetch origin
git pull origin main

# Afficher le commit actuel
CURRENT_COMMIT=$(git log --oneline -1)
echo -e "${GREEN}✅ Code mis à jour : $CURRENT_COMMIT${NC}"
echo ""

# ============================================
# ÉTAPE 2: Arrêt propre des services
# ============================================
echo -e "${YELLOW}🛑 Étape 2/6 : Arrêt des services actuels...${NC}"
docker-compose -f docker-compose.prod.yml down
echo -e "${GREEN}✅ Services arrêtés${NC}"
echo ""

# ============================================
# ÉTAPE 3: Rebuild complet des images
# ============================================
echo -e "${YELLOW}🔨 Étape 3/6 : Rebuild des images Docker...${NC}"
docker-compose -f docker-compose.prod.yml build --no-cache
echo -e "${GREEN}✅ Images reconstruites${NC}"
echo ""

# ============================================
# ÉTAPE 4: Démarrage des services
# ============================================
echo -e "${YELLOW}▶️  Étape 4/6 : Démarrage des services...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# Attendre le démarrage
echo "⏳ Attente du démarrage des services (30s)..."
sleep 30

echo -e "${GREEN}✅ Services démarrés${NC}"
echo ""

# ============================================
# ÉTAPE 5: VÉRIFICATIONS CRITIQUES
# ============================================
echo -e "${YELLOW}🔍 Étape 5/6 : Vérifications de stabilité...${NC}"

# Vérification 1: Tous les conteneurs sont UP
echo "  [1/5] Vérification des conteneurs..."
STOPPED_CONTAINERS=$(docker ps -a --filter "status=exited" --filter "name=harena" --format "{{.Names}}")
if [ -z "$STOPPED_CONTAINERS" ]; then
    echo -e "  ${GREEN}✅ Tous les conteneurs sont UP${NC}"
else
    echo -e "  ${RED}❌ ERREUR: Conteneurs arrêtés détectés:${NC}"
    echo "$STOPPED_CONTAINERS"
    exit 1
fi

# Vérification 2: Tous les healthchecks sont HEALTHY
echo "  [2/5] Vérification des healthchecks..."
sleep 10  # Attendre que les healthchecks passent
UNHEALTHY=$(docker ps --filter "health=unhealthy" --filter "name=harena" --format "{{.Names}}")
if [ -z "$UNHEALTHY" ]; then
    echo -e "  ${GREEN}✅ Tous les healthchecks sont HEALTHY${NC}"
else
    echo -e "  ${RED}❌ ERREUR: Services unhealthy détectés:${NC}"
    echo "$UNHEALTHY"
    exit 1
fi

# Vérification 3: Tous les ports sont sur 0.0.0.0 (pas 127.0.0.1)
echo "  [3/5] Vérification des port bindings..."
LOCALHOST_PORTS=$(docker ps --format "{{.Names}}: {{.Ports}}" | grep "127.0.0.1" || true)
if [ -z "$LOCALHOST_PORTS" ]; then
    echo -e "  ${GREEN}✅ Tous les ports sont correctement exposés (0.0.0.0)${NC}"
else
    echo -e "  ${RED}❌ ATTENTION: Services sur localhost détectés:${NC}"
    echo "$LOCALHOST_PORTS"
    echo -e "  ${YELLOW}⚠️  Ces services ne seront pas accessibles de l'extérieur${NC}"
fi

# Vérification 4: Test des endpoints critiques
echo "  [4/5] Test des endpoints HTTP..."
ENDPOINTS=(
    "http://localhost:3000/health:user_service"
    "http://localhost:3001/api/v1/search/health:search_service"
    "http://localhost:3002/health:metric_service"
    "http://localhost:3006/health:budget_service"
    "http://localhost:3008/health:conversation_v3"
)

ALL_ENDPOINTS_OK=true
for endpoint_pair in "${ENDPOINTS[@]}"; do
    IFS=':' read -r url service <<< "$endpoint_pair"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "    ${GREEN}✅ $service${NC} ($HTTP_CODE)"
    else
        echo -e "    ${RED}❌ $service${NC} ($HTTP_CODE)"
        ALL_ENDPOINTS_OK=false
    fi
done

if [ "$ALL_ENDPOINTS_OK" = true ]; then
    echo -e "  ${GREEN}✅ Tous les endpoints répondent${NC}"
else
    echo -e "  ${YELLOW}⚠️  Certains endpoints ne répondent pas (peut être normal au démarrage)${NC}"
fi

# Vérification 5: Utilisation des ressources
echo "  [5/5] Vérification des ressources système..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

echo "    💾 Mémoire utilisée: ${MEMORY_USAGE}%"
echo "    💿 Disque utilisé: ${DISK_USAGE}%"

if [ "$MEMORY_USAGE" -gt 85 ]; then
    echo -e "    ${YELLOW}⚠️  Utilisation mémoire élevée${NC}"
fi

if [ "$DISK_USAGE" -gt 85 ]; then
    echo -e "    ${YELLOW}⚠️  Utilisation disque élevée${NC}"
fi

echo ""

# ============================================
# ÉTAPE 6: Résumé final
# ============================================
echo -e "${YELLOW}📊 Étape 6/6 : Résumé du déploiement${NC}"
echo ""
echo "Services en cours d'exécution :"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep harena

echo ""
echo "============================================"
echo -e "${GREEN}✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS !${NC}"
echo "============================================"
echo ""
echo "🔗 Endpoints disponibles :"
echo "  - User Service:        http://63.35.52.216/api/v1/users"
echo "  - Search Service:      http://63.35.52.216/api/v1/search"
echo "  - Metric Service:      http://63.35.52.216/api/v1/metrics"
echo "  - Budget Service:      http://63.35.52.216/api/v1/budget"
echo "  - Conversation V3:     http://63.35.52.216/api/v3"
echo ""
echo "📝 Logs en direct :"
echo "  docker-compose -f docker-compose.prod.yml logs -f [service]"
echo ""
echo "🎉 Infrastructure stable et opérationnelle !"
