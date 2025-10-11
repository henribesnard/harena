#!/bin/bash
set -e

# Log tout vers un fichier pour debugging
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=========================================="
echo "Harena EC2 User Data - Début"
echo "Date: $(date)"
echo "=========================================="

# Variables passées par Terraform
DB_HOST="${db_host}"
DB_NAME="${db_name}"
DB_USERNAME="${db_username}"
DB_PASSWORD="${db_password}"
REDIS_ENDPOINT="${redis_endpoint}"
REDIS_AUTH_TOKEN="${redis_auth_token}"
DEEPSEEK_API_KEY="${deepseek_api_key}"
SECRET_KEY="${secret_key}"
ENVIRONMENT="${environment}"

DEPLOY_DIR="/home/ubuntu/harena"
S3_BUCKET="harena-frontend-dev"
AWS_REGION="eu-west-1"

# Étape 1: Mise à jour du système
echo ""
echo "📦 Étape 1/7: Mise à jour du système..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq

# Étape 2: Installation des dépendances système
echo ""
echo "📦 Étape 2/7: Installation des dépendances système..."
apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    postgresql-client \
    redis-tools \
    nginx \
    curl \
    wget \
    unzip \
    awscli

python3.11 --version

# Étape 3: Création de la structure de répertoires
echo ""
echo "📂 Étape 3/7: Création de la structure de répertoires..."
mkdir -p $DEPLOY_DIR
chown ubuntu:ubuntu $DEPLOY_DIR
cd $DEPLOY_DIR

# Créer le virtualenv en tant qu'utilisateur ubuntu
sudo -u ubuntu python3.11 -m venv env

echo "✅ Structure créée"

# Étape 4: Download du code depuis S3
echo ""
echo "⬇️  Étape 4/7: Download du code depuis S3..."

# Attendre que le fichier soit disponible (au cas où le script tourne avant l'upload)
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if aws s3 ls s3://$S3_BUCKET/deploy/harena-deploy.tar.gz --region $AWS_REGION > /dev/null 2>&1; then
        echo "   Archive trouvée dans S3"
        break
    fi
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "⚠️  Archive non trouvée dans S3 après $max_attempts tentatives"
        echo "   Le code devra être déployé manuellement"
        exit 0
    fi
    echo "   Tentative $attempt/$max_attempts - Attente de l'archive..."
    sleep 10
done

# Download de l'archive
sudo -u ubuntu aws s3 cp s3://$S3_BUCKET/deploy/harena-deploy.tar.gz $DEPLOY_DIR/ --region $AWS_REGION --quiet

# Extraction
cd $DEPLOY_DIR
sudo -u ubuntu tar -xzf harena-deploy.tar.gz
rm harena-deploy.tar.gz

echo "✅ Code extrait"

# Étape 5: Configuration de l'environnement
echo ""
echo "🔧 Étape 5/7: Configuration de l'environnement..."

# Créer le fichier .env
cat > $DEPLOY_DIR/.env << ENV_FILE
# Database
DATABASE_URL=postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST/$DB_NAME
POSTGRES_SERVER=$DB_HOST
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USERNAME
POSTGRES_PASSWORD=$DB_PASSWORD

# Redis
REDIS_URL=rediss://default:$REDIS_AUTH_TOKEN@$REDIS_ENDPOINT:6379
REDIS_AUTH_TOKEN=$REDIS_AUTH_TOKEN

# Elasticsearch (local)
SEARCHBOX_URL=http://localhost:9200

# DeepSeek API
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY

# App config
SECRET_KEY=$SECRET_KEY
ENVIRONMENT=$ENVIRONMENT
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=*
ENV_FILE

chown ubuntu:ubuntu $DEPLOY_DIR/.env
chmod 600 $DEPLOY_DIR/.env

echo "✅ Fichier .env créé"

# Étape 6: Installation des dépendances Python
echo ""
echo "🐍 Étape 6/7: Installation des dépendances Python..."
cd $DEPLOY_DIR

# Activer le virtualenv et installer les dépendances
sudo -u ubuntu bash -c "
    source env/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    pip install uvicorn[standard] gunicorn -q
"

echo "   Packages installés:"
sudo -u ubuntu bash -c "source env/bin/activate && pip list | grep -E '(uvicorn|fastapi|sqlalchemy|alembic)'" || true

# Migrations de base de données
echo ""
echo "   Exécution des migrations DB..."
sudo -u ubuntu bash -c "
    source env/bin/activate
    cd conversation_service && alembic upgrade head && cd ..
    cd metric_service && alembic upgrade head && cd ..
" || echo "⚠️  Migrations échouées (peut-être DB pas encore prête)"

echo "✅ Dépendances Python installées"

# Étape 7: Configuration et démarrage des services
echo ""
echo "🚀 Étape 7/7: Configuration et démarrage des services..."

# Créer les fichiers systemd
cat > /etc/systemd/system/conversation-service.service << 'SERVICE_FILE'
[Unit]
Description=Harena Conversation Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/harena
Environment="PATH=/home/ubuntu/harena/env/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ubuntu/harena/.env
ExecStart=/home/ubuntu/harena/env/bin/uvicorn conversation_service.main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_FILE

cat > /etc/systemd/system/metric-service.service << 'SERVICE_FILE'
[Unit]
Description=Harena Metric Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/harena
Environment="PATH=/home/ubuntu/harena/env/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ubuntu/harena/.env
ExecStart=/home/ubuntu/harena/env/bin/uvicorn metric_service.main:app --host 0.0.0.0 --port 8004
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_FILE

# Rechargement et démarrage
systemctl daemon-reload
systemctl enable conversation-service metric-service
systemctl start conversation-service metric-service

echo ""
echo "   Attente démarrage (10s)..."
sleep 10

echo ""
echo "   Statut des services:"
systemctl is-active conversation-service && echo "✅ conversation-service: ACTIF" || echo "❌ conversation-service: INACTIF"
systemctl is-active metric-service && echo "✅ metric-service: ACTIF" || echo "❌ metric-service: INACTIF"

# Installation de CloudWatch Agent
echo ""
echo "📊 Installation de CloudWatch Agent..."
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/arm64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Configuration CloudWatch
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CW_CONFIG'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/user-data.log",
            "log_group_name": "/aws/ec2/harena",
            "log_stream_name": "{instance_id}/user-data"
          },
          {
            "file_path": "/var/log/syslog",
            "log_group_name": "/aws/ec2/harena",
            "log_stream_name": "{instance_id}/syslog"
          }
        ]
      }
    }
  }
}
CW_CONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

echo ""
echo "=========================================="
echo "✅ Setup EC2 terminé avec succès!"
echo "=========================================="
echo ""
echo "Services déployés:"
echo "  - Conversation Service: http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8001"
echo "  - Metric Service: http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8004"
echo ""
echo "Logs disponibles:"
echo "  - User Data: /var/log/user-data.log"
echo "  - Conversation Service: sudo journalctl -u conversation-service -f"
echo "  - Metric Service: sudo journalctl -u metric-service -f"
echo ""
