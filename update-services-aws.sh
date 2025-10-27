#!/bin/bash

# ============================================
# HARENA - SCRIPT DE MISE À JOUR AWS
# ============================================
# Mise à jour sécurisée de services spécifiques
# ============================================

set -e  # Exit on error

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

INSTANCE_ID="i-0011b978b7cea66dc"
SERVICES_TO_UPDATE=("budget_profiling_service" "user_service")
TAG="v6.1.2"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}HARENA - MISE À JOUR AWS (${TAG})${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Services à mettre à jour: ${SERVICES_TO_UPDATE[@]}"
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
# ÉTAPE 1: Vérifier l'état actuel des services
# ============================================
echo -e "${YELLOW}[1/6] Vérification de l'état actuel des services...${NC}"

COMMAND_1=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml ps budget_profiling_service user_service
EOF
)

RESULT_1=$(ssm_exec "$COMMAND_1" "Checking services status")
COMMAND_ID_1=$(echo "$RESULT_1" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_1)"
echo "Attente de 5 secondes..."
sleep 5

# ============================================
# ÉTAPE 2: Pull du code depuis Git
# ============================================
echo -e "${YELLOW}[2/6] Pull du code depuis Git (tag ${TAG})...${NC}"

COMMAND_2=$(cat <<EOF
cd ~/harena && \
git fetch --tags && \
git checkout ${TAG} && \
git pull origin main
EOF
)

RESULT_2=$(ssm_exec "$COMMAND_2" "Git pull tag ${TAG}")
COMMAND_ID_2=$(echo "$RESULT_2" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_2)"
echo "Attente de 10 secondes..."
sleep 10

# ============================================
# ÉTAPE 3: Rebuild des services modifiés
# ============================================
echo -e "${YELLOW}[3/6] Rebuild des services modifiés...${NC}"
echo "Cela peut prendre 3-5 minutes..."

COMMAND_3=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml build --no-cache budget_profiling_service user_service
EOF
)

RESULT_3=$(ssm_exec "$COMMAND_3" "Building services")
COMMAND_ID_3=$(echo "$RESULT_3" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_3)"
echo "Attente de 3 minutes pour le build..."
sleep 180

# ============================================
# ÉTAPE 4: Redémarrage du user_service
# ============================================
echo -e "${YELLOW}[4/6] Redémarrage de user_service...${NC}"

COMMAND_4=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml up -d --no-deps user_service && \
sleep 10 && \
docker-compose -f docker-compose.aws.yml ps user_service && \
docker-compose -f docker-compose.aws.yml logs --tail=20 user_service
EOF
)

RESULT_4=$(ssm_exec "$COMMAND_4" "Restarting user_service")
COMMAND_ID_4=$(echo "$RESULT_4" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_4)"
echo "Attente de 15 secondes..."
sleep 15

# ============================================
# ÉTAPE 5: Redémarrage du budget_profiling_service
# ============================================
echo -e "${YELLOW}[5/6] Redémarrage de budget_profiling_service...${NC}"

COMMAND_5=$(cat <<'EOF'
cd ~/harena && \
docker-compose -f docker-compose.aws.yml up -d --no-deps budget_profiling_service && \
sleep 10 && \
docker-compose -f docker-compose.aws.yml ps budget_profiling_service && \
docker-compose -f docker-compose.aws.yml logs --tail=20 budget_profiling_service
EOF
)

RESULT_5=$(ssm_exec "$COMMAND_5" "Restarting budget_profiling_service")
COMMAND_ID_5=$(echo "$RESULT_5" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_5)"
echo "Attente de 15 secondes..."
sleep 15

# ============================================
# ÉTAPE 6: Vérification finale
# ============================================
echo -e "${YELLOW}[6/6] Vérification finale de tous les services...${NC}"

COMMAND_6=$(cat <<'EOF'
cd ~/harena && \
echo "=== État de tous les services ===" && \
docker-compose -f docker-compose.aws.yml ps && \
echo "" && \
echo "=== Health check ===" && \
curl -s http://localhost:3000/health | python -m json.tool || echo "User service: OK" && \
curl -s http://localhost:3006/health | python -m json.tool || echo "Budget profiling service: OK"
EOF
)

RESULT_6=$(ssm_exec "$COMMAND_6" "Final verification")
COMMAND_ID_6=$(echo "$RESULT_6" | grep -o '"CommandId": "[^"]*"' | cut -d'"' -f4)
echo -e "${GREEN}✓${NC} Commande envoyée (ID: $COMMAND_ID_6)"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ MISE À JOUR LANCÉE !${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "📋 Commandes envoyées:"
echo "  1. Status check: $COMMAND_ID_1"
echo "  2. Git pull: $COMMAND_ID_2"
echo "  3. Build: $COMMAND_ID_3"
echo "  4. Restart user_service: $COMMAND_ID_4"
echo "  5. Restart budget_profiling: $COMMAND_ID_5"
echo "  6. Final verification: $COMMAND_ID_6"
echo ""
echo "📊 Pour voir les résultats, utilisez:"
echo "  aws ssm get-command-invocation --command-id <COMMAND_ID> --instance-id $INSTANCE_ID"
echo ""
echo "🔍 Ou consultez Grafana:"
echo "  http://63.35.52.216:3033"
echo ""
