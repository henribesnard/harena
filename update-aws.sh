#!/bin/bash

# ============================================
# HARENA - SCRIPT DE MISE À JOUR AWS
# ============================================
# Mise à jour de la stack sans downtime prolongé
# ============================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
EC2_IP="${EC2_IP:-63.35.52.216}"
EC2_USER="${EC2_USER:-ec2-user}"
SSH_KEY="${SSH_KEY:-~/.ssh/harena-aws.pem}"
COMPOSE_FILE="docker-compose.aws.yml"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}HARENA - MISE À JOUR AWS${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Vérification des arguments
SERVICE="$1"
if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service_name|all>"
    echo ""
    echo "Services disponibles:"
    echo "  • user_service"
    echo "  • search_service"
    echo "  • metric_service"
    echo "  • conversation_service_v3"
    echo "  • budget_profiling_service"
    echo "  • frontend"
    echo "  • nginx"
    echo "  • all (tous les services)"
    exit 1
fi

# Fonction pour exécuter des commandes SSH
ssh_exec() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" "$@"
}

# Fonction pour copier des fichiers
scp_copy() {
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r "$1" "$EC2_USER@$EC2_IP:$2"
}

echo -e "${YELLOW}Service à mettre à jour: ${SERVICE}${NC}"
echo ""

# ============================================
# ÉTAPE 1: Backup avant mise à jour
# ============================================
echo -e "${YELLOW}[1/4] Création d'un backup de sécurité...${NC}"
ssh_exec << 'EOF'
    cd ~/harena
    DATE=$(date +%Y%m%d_%H%M%S)

    # Backup de la base de données
    docker exec harena_postgres pg_dump -U harena_admin harena | gzip > ~/harena/backups/harena_pre_update_$DATE.sql.gz

    echo "Backup créé: harena_pre_update_$DATE.sql.gz"
EOF
echo -e "${GREEN}✓${NC} Backup créé"

# ============================================
# ÉTAPE 2: Copie du nouveau code
# ============================================
echo -e "${YELLOW}[2/4] Copie du nouveau code...${NC}"

if [ "$SERVICE" = "all" ]; then
    # Mise à jour de tous les services
    SERVICES=(
        "user_service"
        "search_service"
        "metric_service"
        "conversation_service_v3"
        "budget_profiling_service"
        "harena_front"
    )

    for svc in "${SERVICES[@]}"; do
        if [ -d "$svc" ]; then
            scp_copy "$svc" "~/harena/"
            echo -e "${GREEN}  ✓${NC} $svc copié"
        fi
    done

    # Copie docker-compose et configs
    scp_copy "$COMPOSE_FILE" "~/harena/"
    scp_copy "nginx/nginx.conf" "~/harena/nginx/"
    scp_copy "nginx/conf.d/harena.conf" "~/harena/nginx/conf.d/"

else
    # Mise à jour d'un service spécifique
    if [ -d "$SERVICE" ]; then
        scp_copy "$SERVICE" "~/harena/"
        echo -e "${GREEN}  ✓${NC} $SERVICE copié"
    elif [ "$SERVICE" = "nginx" ]; then
        scp_copy "nginx/nginx.conf" "~/harena/nginx/"
        scp_copy "nginx/conf.d/harena.conf" "~/harena/nginx/conf.d/"
        echo -e "${GREEN}  ✓${NC} Configuration Nginx copiée"
    else
        echo -e "${RED}❌ Service non trouvé: $SERVICE${NC}"
        exit 1
    fi
fi

# ============================================
# ÉTAPE 3: Rebuild et redémarrage
# ============================================
echo -e "${YELLOW}[3/4] Rebuild et redémarrage...${NC}"

if [ "$SERVICE" = "all" ]; then
    ssh_exec << 'EOF'
        cd ~/harena

        echo "Rebuild de tous les services..."
        docker-compose -f docker-compose.aws.yml build

        echo "Redémarrage progressif..."
        docker-compose -f docker-compose.aws.yml up -d

        echo "Attente de stabilisation (30s)..."
        sleep 30
EOF
else
    ssh_exec << EOF
        cd ~/harena

        echo "Rebuild de $SERVICE..."
        docker-compose -f docker-compose.aws.yml build $SERVICE

        echo "Redémarrage de $SERVICE..."
        docker-compose -f docker-compose.aws.yml up -d --no-deps $SERVICE

        echo "Attente de stabilisation (15s)..."
        sleep 15
EOF
fi

echo -e "${GREEN}✓${NC} Services redémarrés"

# ============================================
# ÉTAPE 4: Vérification
# ============================================
echo -e "${YELLOW}[4/4] Vérification de l'état...${NC}"
ssh_exec << 'EOF'
    cd ~/harena
    docker-compose -f docker-compose.aws.yml ps
EOF

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ MISE À JOUR TERMINÉE !${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Vérification de santé
echo -e "${YELLOW}Test de santé...${NC}"
sleep 5

if curl -s -o /dev/null -w "%{http_code}" "http://$EC2_IP/health" | grep -q "200"; then
    echo -e "${GREEN}✓${NC} Application: OK"
else
    echo -e "${RED}⚠${NC} Application: Vérifiez les logs"
    echo "  ssh -i $SSH_KEY $EC2_USER@$EC2_IP"
    echo "  cd ~/harena && docker-compose -f docker-compose.aws.yml logs -f $SERVICE"
fi

echo ""
echo "📊 Voir les logs:"
echo "  ssh -i $SSH_KEY $EC2_USER@$EC2_IP"
echo "  cd ~/harena"
echo "  docker-compose -f docker-compose.aws.yml logs -f $SERVICE"
