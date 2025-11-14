#!/bin/bash
# Script pour corriger le DATABASE_URL sur AWS

EC2_IP="63.35.52.216"
EC2_USER="ec2-user"
HARENA_DIR="/home/ec2-user/harena"
SSH_KEY="~/.ssh/harena-deploy-key.pem"

echo "🔧 Correction du DATABASE_URL sur AWS..."
echo "=========================================="

# Connexion SSH et vérification + correction
ssh -i $SSH_KEY $EC2_USER@$EC2_IP << 'EOF'
cd /home/ec2-user/harena

echo "📋 DATABASE_URL actuel:"
grep "DATABASE_URL" .env

echo ""
echo "🔄 Correction du DATABASE_URL..."

# Remplacer harena-postgres par l'IP
sed -i 's/@harena-postgres:/@63.35.52.216:/g' .env

echo ""
echo "✅ Nouveau DATABASE_URL:"
grep "DATABASE_URL" .env

echo ""
echo "♻️  Redémarrage du sync_service..."
docker-compose restart sync_service

echo ""
echo "⏳ Attente du démarrage (5 secondes)..."
sleep 5

echo ""
echo "📊 Statut du service:"
docker ps | grep sync_service

echo ""
echo "📝 Logs récents:"
docker logs harena_sync_service --tail 20

EOF

echo ""
echo "=========================================="
echo "✅ Configuration terminée !"
echo ""
echo "🧪 Test du webhook:"
curl -X POST http://$EC2_IP:3004/webhooks/bridge \
  -H "Content-Type: application/json" \
  -d '{"test": "after_fix", "type": "test"}' \
  | jq .

echo ""
echo "📋 Si le test affiche 'status: received' sans erreur, c'est bon !"
