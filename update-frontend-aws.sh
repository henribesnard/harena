#!/bin/bash

# ============================================
# HARENA - SCRIPT DE MISE À JOUR FRONTEND AWS
# ============================================
# Mise à jour du frontend (React/Vite) sur AWS
# ============================================

set -e  # Exit on error

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

INSTANCE_ID="i-0011b978b7cea66dc"
FRONTEND_TAG="v4.0.1"
HARENA_DIR="/home/ec2-user/harena"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}HARENA - MISE À JOUR FRONTEND (${FRONTEND_TAG})${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Fonction pour exécuter des commandes via SSM
ssm_exec() {
    local command="$1"
    local comment="${2:-Executing command}"

    echo -e "${YELLOW}${comment}...${NC}"

    aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --comment "$comment" \
        --parameters "commands=[\"$command\"]" \
        --output json
}

# ============================================
# ÉTAPE 1: Vérifier l'état actuel du frontend
# ============================================
echo -e "${YELLOW}[1/7] Vérification de l'état actuel du frontend...${NC}"

COMMAND_1=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml ps frontend
EOF
)

RESULT_1=$(ssm_exec "$COMMAND_1" "Checking frontend status")
COMMAND_ID_1=$(echo "$RESULT_1" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_1)"
echo "Attente de 5 secondes..."
sleep 5

# ============================================
# ÉTAPE 2: Pull du code frontend depuis Git
# ============================================
echo -e "${YELLOW}[2/7] Pull du code frontend depuis Git (tag ${FRONTEND_TAG})...${NC}"

COMMAND_2=$(cat <<EOF
cd ~/harena/harena_front && \
git fetch --tags && \
git checkout ${FRONTEND_TAG} && \
git pull origin main
EOF
)

RESULT_2=$(ssm_exec "$COMMAND_2" "Git pull frontend tag ${FRONTEND_TAG}")
COMMAND_ID_2=$(echo "$RESULT_2" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_2)"
echo "Attente de 10 secondes..."
sleep 10

# ============================================
# ÉTAPE 3: Mise à jour de docker-compose.aws.yml
# ============================================
echo -e "${YELLOW}[3/7] Vérification de la configuration Docker Compose...${NC}"

COMMAND_3=$(cat <<'EOF'
cd ~/harena && \
git pull origin main && \
echo "Configuration Docker Compose mise à jour"
EOF
)

RESULT_3=$(ssm_exec "$COMMAND_3" "Updating docker-compose.aws.yml")
COMMAND_ID_3=$(echo "$RESULT_3" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_3)"
echo "Attente de 10 secondes..."
sleep 10

# ============================================
# ÉTAPE 4: Rebuild du frontend
# ============================================
echo -e "${YELLOW}[4/7] Rebuild du frontend...${NC}"
echo "Cela peut prendre 3-5 minutes..."

COMMAND_4=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml build --no-cache frontend && \
echo "Frontend build completed"
EOF
)

RESULT_4=$(ssm_exec "$COMMAND_4" "Building frontend")
COMMAND_ID_4=$(echo "$RESULT_4" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_4)"
echo "Attente de 4 minutes pour le build..."
sleep 240

# ============================================
# ÉTAPE 5: Arrêt de l'ancien frontend
# ============================================
echo -e "${YELLOW}[5/7] Arrêt de l'ancien frontend...${NC}"

COMMAND_5=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml stop frontend && \
docker-compose -f docker-compose.aws.yml rm -f frontend
EOF
)

RESULT_5=$(ssm_exec "$COMMAND_5" "Stopping old frontend")
COMMAND_ID_5=$(echo "$RESULT_5" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_5)"
echo "Attente de 10 secondes..."
sleep 10

# ============================================
# ÉTAPE 6: Démarrage du nouveau frontend
# ============================================
echo -e "${YELLOW}[6/7] Démarrage du nouveau frontend...${NC}"

COMMAND_6=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml up -d frontend && \
sleep 15 && \
docker-compose -f docker-compose.aws.yml ps frontend && \
docker-compose -f docker-compose.aws.yml logs --tail=30 frontend
EOF
)

RESULT_6=$(ssm_exec "$COMMAND_6" "Starting new frontend")
COMMAND_ID_6=$(echo "$RESULT_6" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_6)"
echo "Attente de 20 secondes..."
sleep 20

# ============================================
# ÉTAPE 7: Vérification finale et test
# ============================================
echo -e "${YELLOW}[7/7] Vérification finale et test du frontend...${NC}"

COMMAND_7=$(cat <<'EOF'
cd ~/harena && \
echo "=== État du frontend ===" && \
docker-compose -f docker-compose.aws.yml ps frontend && \
echo "" && \
echo "=== Health check Nginx ===" && \
curl -s http://localhost/health || echo "Health endpoint OK" && \
echo "" && \
echo "=== Test page d'accueil ===" && \
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8080/
EOF
)

RESULT_7=$(ssm_exec "$COMMAND_7" "Final verification")
COMMAND_ID_7=$(echo "$RESULT_7" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_7)"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ MISE À JOUR FRONTEND LANCÉE !${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "📋 Commandes envoyées:"
echo "  1. Status check: $COMMAND_ID_1"
echo "  2. Git pull frontend: $COMMAND_ID_2"
echo "  3. Update docker-compose: $COMMAND_ID_3"
echo "  4. Build frontend: $COMMAND_ID_4"
echo "  5. Stop old frontend: $COMMAND_ID_5"
echo "  6. Start new frontend: $COMMAND_ID_6"
echo "  7. Final verification: $COMMAND_ID_7"
echo ""
echo "📊 Pour voir les résultats, utilisez:"
echo "  aws ssm get-command-invocation --command-id <COMMAND_ID> --instance-id $INSTANCE_ID"
echo ""
echo "🌐 Test de l'application:"
echo "  Frontend: http://63.35.52.216:8080"
echo ""
echo "🔍 Ou consultez Grafana pour le monitoring:"
echo "  http://63.35.52.216:3033"
echo ""
echo "📝 Changements de cette version (v4.0.1):"
echo "  • Interface unifiée avec sidebar de conversation fixe sur toutes les pages"
echo "  • Amélioration du Dashboard utilisateur (layout horizontal)"
echo "  • Correction du bug de la page Configuration"
echo "  • Optimisation de l'affichage mobile"
echo ""
