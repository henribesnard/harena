#!/bin/bash
set -e

# User data pour EC2 All-in-One
# Installe Docker, Docker Compose et déploie l'application

echo "=========================================="
echo "Harena All-in-One Setup"
echo "=========================================="

# Mise à jour du système
apt-get update
apt-get upgrade -y

# Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# Installer Docker Compose
DOCKER_COMPOSE_VERSION="2.24.0"
curl -L "https://github.com/docker/compose/releases/download/v$DOCKER_COMPOSE_VERSION/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Installer AWS CLI
apt-get install -y awscli

# Créer le répertoire de l'application
mkdir -p /opt/harena
cd /opt/harena

# Créer le fichier .env
cat > .env <<EOF
# Database
DB_NAME=${db_name}
DB_USERNAME=${db_username}
DB_PASSWORD=${db_password}

# Redis
REDIS_AUTH_TOKEN=${redis_auth_token}

# API
SECRET_KEY=${secret_key}
DEEPSEEK_API_KEY=${deepseek_api_key}

# Environment
ENVIRONMENT=${environment}

# Backup
BACKUP_S3_BUCKET=harena-backups-${environment}
EOF

chmod 600 .env

# Créer le répertoire pour les backups
mkdir -p /opt/harena/migration/backups

# Créer un script d'installation qui attend le dépôt Git
cat > /opt/harena/install.sh <<'INSTALL_SCRIPT'
#!/bin/bash
set -e

echo "Waiting for application code to be deployed..."
echo "Please run: ./migration/2_deploy_new_infrastructure.sh"
echo ""
echo "This instance is ready to receive the application."
echo "IP Address: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
INSTALL_SCRIPT

chmod +x /opt/harena/install.sh

# Message de fin
echo "=========================================="
echo "Instance All-in-One configurée"
echo "=========================================="
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker-compose --version)"
echo ""
echo "Prochaine étape: Déployer le code de l'application"
