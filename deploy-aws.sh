#!/bin/bash

# ============================================
# HARENA - SCRIPT DE DÉPLOIEMENT AWS
# ============================================
# Déploiement initial de la stack complète sur EC2
# ============================================

set -e  # Exit on error

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EC2_IP="${EC2_IP:-63.35.52.216}"
EC2_USER="${EC2_USER:-ec2-user}"
SSH_KEY="${SSH_KEY:-~/.ssh/harena-aws.pem}"
COMPOSE_FILE="docker-compose.aws.yml"
ENV_FILE=".env.production"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}HARENA - DÉPLOIEMENT AWS${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Vérification de la clé SSH
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ Clé SSH non trouvée: $SSH_KEY${NC}"
    echo -e "${YELLOW}💡 Spécifiez le chemin avec: export SSH_KEY=/path/to/key.pem${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration:"
echo "  • EC2 IP: $EC2_IP"
echo "  • User: $EC2_USER"
echo "  • SSH Key: $SSH_KEY"
echo ""

# Fonction pour exécuter des commandes SSH
ssh_exec() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" "$@"
}

# Fonction pour copier des fichiers vers EC2
scp_copy() {
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r "$1" "$EC2_USER@$EC2_IP:$2"
}

# ============================================
# ÉTAPE 1: Vérification de la connexion
# ============================================
echo -e "${YELLOW}[1/8] Vérification de la connexion SSH...${NC}"
if ssh_exec "echo 'Connexion OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Connexion SSH établie"
else
    echo -e "${RED}❌ Impossible de se connecter à l'EC2${NC}"
    exit 1
fi

# ============================================
# ÉTAPE 2: Installation des dépendances
# ============================================
echo -e "${YELLOW}[2/8] Installation de Docker et Docker Compose...${NC}"
ssh_exec << 'EOF'
    # Mise à jour système
    sudo yum update -y

    # Installation Docker si pas déjà installé
    if ! command -v docker &> /dev/null; then
        echo "Installation de Docker..."
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    else
        echo "Docker déjà installé"
    fi

    # Installation Docker Compose si pas déjà installé
    if ! command -v docker-compose &> /dev/null; then
        echo "Installation de Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Docker Compose déjà installé"
    fi

    # Vérification
    docker --version
    docker-compose --version
EOF
echo -e "${GREEN}✓${NC} Docker et Docker Compose installés"

# ============================================
# ÉTAPE 3: Création de la structure de dossiers
# ============================================
echo -e "${YELLOW}[3/8] Création de la structure de dossiers...${NC}"
ssh_exec << 'EOF'
    mkdir -p ~/harena
    mkdir -p ~/harena/nginx/conf.d
    mkdir -p ~/harena/nginx/ssl
    mkdir -p ~/harena/nginx/logs
    mkdir -p ~/harena/monitoring/prometheus
    mkdir -p ~/harena/monitoring/loki
    mkdir -p ~/harena/monitoring/promtail
    mkdir -p ~/harena/monitoring/grafana/datasources
    mkdir -p ~/harena/monitoring/grafana/dashboards/json
    mkdir -p ~/harena/backups
    mkdir -p ~/harena/logs
EOF
echo -e "${GREEN}✓${NC} Structure de dossiers créée"

# ============================================
# ÉTAPE 4: Copie des fichiers de configuration
# ============================================
echo -e "${YELLOW}[4/8] Copie des fichiers de configuration...${NC}"

# Docker Compose
scp_copy "$COMPOSE_FILE" "~/harena/"
echo -e "${GREEN}  ✓${NC} docker-compose.aws.yml copié"

# Nginx
scp_copy "nginx/nginx.conf" "~/harena/nginx/"
scp_copy "nginx/conf.d/harena.conf" "~/harena/nginx/conf.d/"
echo -e "${GREEN}  ✓${NC} Configuration Nginx copiée"

# Monitoring
scp_copy "monitoring/prometheus/prometheus.yml" "~/harena/monitoring/prometheus/"
scp_copy "monitoring/loki/loki-config.yml" "~/harena/monitoring/loki/"
scp_copy "monitoring/promtail/promtail-config.yml" "~/harena/monitoring/promtail/"
scp_copy "monitoring/grafana/datasources/datasources.yml" "~/harena/monitoring/grafana/datasources/"
scp_copy "monitoring/grafana/dashboards/dashboards.yml" "~/harena/monitoring/grafana/dashboards/"
echo -e "${GREEN}  ✓${NC} Configuration monitoring copiée"

# Fichier .env
if [ -f "$ENV_FILE" ]; then
    scp_copy "$ENV_FILE" "~/harena/.env"
    echo -e "${GREEN}  ✓${NC} Fichier .env copié"
else
    echo -e "${YELLOW}  ⚠${NC} Fichier $ENV_FILE non trouvé, vous devrez le créer manuellement"
fi

# ============================================
# ÉTAPE 5: Copie du code source
# ============================================
echo -e "${YELLOW}[5/8] Copie du code source...${NC}"
echo "Cela peut prendre quelques minutes..."

# Liste des services à copier
SERVICES=(
    "user_service"
    "search_service"
    "metric_service"
    "conversation_service_v3"
    "budget_profiling_service"
    "db_service"
    "config_service"
    "harena_front"
)

for service in "${SERVICES[@]}"; do
    if [ -d "$service" ]; then
        scp_copy "$service" "~/harena/"
        echo -e "${GREEN}  ✓${NC} $service copié"
    else
        echo -e "${YELLOW}  ⚠${NC} $service non trouvé (ignoré)"
    fi
done

# ============================================
# ÉTAPE 6: Configuration du Swap (2GB)
# ============================================
echo -e "${YELLOW}[6/8] Configuration du swap (2GB)...${NC}"
ssh_exec << 'EOF'
    # Vérifier si swap existe déjà
    if [ ! -f /swapfile ]; then
        echo "Création du fichier swap..."
        sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile

        # Rendre permanent
        if ! grep -q '/swapfile' /etc/fstab; then
            echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
        fi

        # Optimiser swappiness
        sudo sysctl vm.swappiness=10
        if ! grep -q 'vm.swappiness' /etc/sysctl.conf; then
            echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
        fi

        echo "Swap configuré avec succès"
    else
        echo "Swap déjà configuré"
    fi

    # Vérification
    free -h
EOF
echo -e "${GREEN}✓${NC} Swap configuré"

# ============================================
# ÉTAPE 7: Build des images Docker
# ============================================
echo -e "${YELLOW}[7/8] Build des images Docker...${NC}"
echo "Cela peut prendre 10-15 minutes..."

ssh_exec << 'EOF'
    cd ~/harena
    docker-compose -f docker-compose.aws.yml build
EOF
echo -e "${GREEN}✓${NC} Images Docker buildées"

# ============================================
# ÉTAPE 8: Lancement de la stack
# ============================================
echo -e "${YELLOW}[8/8] Lancement de la stack Docker Compose...${NC}"
ssh_exec << 'EOF'
    cd ~/harena
    docker-compose -f docker-compose.aws.yml up -d

    echo ""
    echo "Attente du démarrage des services (30s)..."
    sleep 30

    echo ""
    echo "État des containers:"
    docker-compose -f docker-compose.aws.yml ps
EOF

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ DÉPLOIEMENT TERMINÉ !${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "🌐 Accès aux services:"
echo "  • Frontend:         http://$EC2_IP"
echo "  • API (v1):         http://$EC2_IP/api/v1/"
echo "  • API (v3):         http://$EC2_IP/api/v3/"
echo "  • Grafana:          http://$EC2_IP:3033 (admin/HarenaAdmin2024!)"
echo ""
echo "📊 Vérifier les logs:"
echo "  ssh -i $SSH_KEY $EC2_USER@$EC2_IP"
echo "  cd ~/harena"
echo "  docker-compose -f docker-compose.aws.yml logs -f"
echo ""
echo "🔍 Monitoring:"
echo "  • Prometheus:       http://$EC2_IP:9090"
echo "  • Grafana:          http://$EC2_IP:3033"
echo ""

# Test de santé
echo -e "${YELLOW}Test de santé des services...${NC}"
sleep 5

if curl -s -o /dev/null -w "%{http_code}" "http://$EC2_IP/health" | grep -q "200"; then
    echo -e "${GREEN}✓${NC} Nginx: OK"
else
    echo -e "${YELLOW}⚠${NC} Nginx: En cours de démarrage..."
fi

echo ""
echo -e "${GREEN}Déploiement réussi ! 🚀${NC}"
