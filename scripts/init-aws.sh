#!/bin/bash
set -e

echo "🚀 Initialisation AWS pour Harena"
echo "=================================="

# Load environment variables (safe method)
if [ ! -f .env ]; then
    echo "❌ Fichier .env introuvable"
    exit 1
fi

# Extract specific variables using grep
AWS_REGION=$(grep '^AWS_REGION=' .env | cut -d'=' -f2)
AWS_ACCOUNT_ID=$(grep '^AWS_ACCOUNT_ID=' .env | cut -d'=' -f2)
AWS_ACCESS_KEY_ID=$(grep '^AWS_ACCESS_KEY_ID=' .env | cut -d'=' -f2)
AWS_SECRET_ACCESS_KEY=$(grep '^AWS_SECRET_ACCESS_KEY=' .env | cut -d'=' -f2)
AWS_DB_PASSWORD=$(grep '^AWS_DB_PASSWORD=' .env | cut -d'=' -f2-)
AWS_REDIS_AUTH_TOKEN=$(grep '^AWS_REDIS_AUTH_TOKEN=' .env | cut -d'=' -f2-)
DEEPSEEK_API_KEY=$(grep '^DEEPSEEK_API_KEY=' .env | cut -d'=' -f2)
SECRET_KEY=$(grep '^SECRET_KEY=' .env | cut -d'=' -f2)
ALERT_EMAIL=$(grep '^ALERT_EMAIL=' .env | cut -d'=' -f2)
DOMAIN_NAME=$(grep '^DOMAIN_NAME=' .env | cut -d'=' -f2)
ENVIRONMENT=$(grep '^ENVIRONMENT=' .env | cut -d'=' -f2)

# Check AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ AWS credentials manquantes dans .env"
    exit 1
fi

# Configure AWS CLI
echo "📝 Configuration AWS CLI..."
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_REGION"
aws configure set output json

# Verify AWS connection
echo "✅ Vérification de la connexion AWS..."
aws sts get-caller-identity

# Create terraform.tfvars from .env
echo "📝 Création de terraform/terraform.tfvars..."
cat > terraform/terraform.tfvars << EOF
# AWS Configuration
aws_region     = "$AWS_REGION"
environment    = "$ENVIRONMENT"
aws_account_id = "$AWS_ACCOUNT_ID"

# Network
vpc_cidr = "10.0.0.0/16"

# RDS PostgreSQL
rds_instance_class    = "db.t4g.micro"
rds_allocated_storage = 20
db_name              = "harena"
db_username          = "harena_admin"
db_password          = "$AWS_DB_PASSWORD"

# ElastiCache Redis
redis_node_type  = "cache.t4g.micro"
redis_auth_token = "$AWS_REDIS_AUTH_TOKEN"

# EC2
ec2_instance_type  = "t4g.micro"
use_spot_instances = true
spot_max_price     = "0.0042"
ebs_volume_size    = 20

# Application secrets
deepseek_api_key = "$DEEPSEEK_API_KEY"
secret_key       = "$SECRET_KEY"

# Frontend
domain_name = "$DOMAIN_NAME"

# Monitoring
enable_auto_shutdown = true
shutdown_night      = true
shutdown_weekend    = true
alert_email         = "$ALERT_EMAIL"
EOF

echo "✅ terraform.tfvars créé"

# Initialize Terraform
echo "🔧 Initialisation Terraform..."
cd terraform
terraform init

echo ""
echo "✅ Initialisation terminée !"
echo ""
echo "📋 Prochaines étapes :"
echo "  1. cd terraform"
echo "  2. terraform plan    # Vérifier le plan"
echo "  3. terraform apply   # Créer l'infrastructure"
echo ""
echo "⏱️  Durée estimée : 15-20 minutes"
echo "💰 Coût estimé : ~14-20 EUR/mois"
