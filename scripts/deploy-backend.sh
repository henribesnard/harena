#!/bin/bash
set -e

echo "🚀 Déploiement Backend Harena"
echo "=============================="

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Fichier .env introuvable"
    exit 1
fi

# Get EC2 instance ID from Terraform
cd terraform
EC2_INSTANCE_ID=$(terraform output -raw ec2_instance_id 2>/dev/null || echo "")
EC2_PUBLIC_IP=$(terraform output -raw ec2_public_ip 2>/dev/null || echo "")
cd ..

if [ -z "$EC2_INSTANCE_ID" ]; then
    echo "❌ Instance EC2 introuvable. Lancez d'abord 'terraform apply'"
    exit 1
fi

echo "📦 Instance EC2: $EC2_INSTANCE_ID"
echo "🌐 IP publique: $EC2_PUBLIC_IP"

# Build Docker image
echo "🐳 Construction de l'image Docker..."
docker build -t harena-backend:latest .

# Tag for ECR or Docker Hub (adjust as needed)
DOCKER_IMAGE="harena-backend:latest"

# Save image to tar
echo "💾 Export de l'image Docker..."
docker save $DOCKER_IMAGE | gzip > /tmp/harena-backend.tar.gz

# Copy to EC2
echo "📤 Transfert de l'image vers EC2..."
aws s3 cp /tmp/harena-backend.tar.gz s3://harena-deployments-temp/backend.tar.gz

# SSH into EC2 and deploy
echo "🚢 Déploiement sur EC2..."
aws ssm send-command \
    --instance-ids "$EC2_INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "aws s3 cp s3://harena-deployments-temp/backend.tar.gz /tmp/backend.tar.gz",
        "cd /opt/harena",
        "docker load < /tmp/backend.tar.gz",
        "docker-compose down backend || true",
        "docker-compose up -d backend",
        "rm /tmp/backend.tar.gz"
    ]' \
    --region "$AWS_REGION"

echo "⏳ Attente du démarrage de l'application (30s)..."
sleep 30

# Run database migrations
echo "🗄️  Exécution des migrations..."
aws ssm send-command \
    --instance-ids "$EC2_INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=[
        "cd /opt/harena",
        "docker-compose exec -T backend alembic upgrade head"
    ]' \
    --region "$AWS_REGION"

# Health check
echo "🏥 Vérification du health check..."
for i in {1..10}; do
    if curl -f "http://$EC2_PUBLIC_IP:8000/health" 2>/dev/null; then
        echo "✅ Backend déployé avec succès !"
        echo "🔗 URL: http://$EC2_PUBLIC_IP:8000"
        exit 0
    fi
    echo "   Tentative $i/10..."
    sleep 3
done

echo "⚠️  Le backend ne répond pas aux health checks"
echo "   Vérifiez les logs: aws ssm start-session --target $EC2_INSTANCE_ID"
exit 1
