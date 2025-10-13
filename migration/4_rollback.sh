#!/bin/bash
set -e

# Script de rollback en cas de problème avec la migration
# Usage: ./4_rollback.sh <backup_directory>

if [ $# -lt 1 ]; then
  echo "Usage: $0 <backup_directory>"
  echo ""
  echo "Example:"
  echo "  $0 ./migration/backups/20250112_143000"
  exit 1
fi

BACKUP_DIR=$1

if [ ! -d "$BACKUP_DIR" ]; then
  echo "❌ ERREUR: Le répertoire de backup n'existe pas: $BACKUP_DIR"
  exit 1
fi

echo "=========================================="
echo "ROLLBACK DE LA MIGRATION"
echo "=========================================="
echo "Backup source: $BACKUP_DIR"
echo ""
echo "⚠️  ATTENTION ⚠️"
echo "Ce script va:"
echo "  1. Redémarrer l'ancienne infrastructure (RDS, Redis, EC2)"
echo "  2. Restaurer les données depuis le backup"
echo "  3. Arrêter la nouvelle infrastructure"
echo ""

read -p "Continuer avec le rollback? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Rollback annulé"
  exit 0
fi

echo ""

# Récupérer les informations de l'ancienne infrastructure
echo "1. Récupération de l'ancienne infrastructure..."

OLD_INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=harena-backend-dev" "Name=instance-state-name,Values=stopped,running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ "$OLD_INSTANCE_ID" = "None" ] || [ -z "$OLD_INSTANCE_ID" ]; then
  echo "❌ ERREUR: Ancienne instance EC2 non trouvée"
  echo "   L'infrastructure a peut-être déjà été détruite."
  echo "   Impossible de faire un rollback automatique."
  exit 1
fi

echo "   Ancienne instance: $OLD_INSTANCE_ID"

# Redémarrer l'ancienne instance si nécessaire
INSTANCE_STATE=$(aws ec2 describe-instances \
  --instance-ids $OLD_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)

if [ "$INSTANCE_STATE" = "stopped" ]; then
  echo "   Démarrage de l'ancienne instance..."
  aws ec2 start-instances --instance-ids $OLD_INSTANCE_ID

  echo "   Attente du démarrage..."
  aws ec2 wait instance-running --instance-ids $OLD_INSTANCE_ID
  sleep 30

  echo "   ✓ Instance démarrée"
fi

OLD_INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $OLD_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "   IP de l'ancienne instance: $OLD_INSTANCE_IP"
echo ""

# Vérifier RDS
echo "2. Vérification de RDS..."
RDS_STATUS=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].DBInstanceStatus' \
  --output text 2>/dev/null || echo "not-found")

if [ "$RDS_STATUS" = "not-found" ]; then
  echo "❌ ERREUR: RDS n'existe plus"
  echo "   Impossible de faire un rollback automatique."
  exit 1
elif [ "$RDS_STATUS" = "stopped" ]; then
  echo "   Démarrage de RDS..."
  aws rds start-db-instance --db-instance-identifier harena-db-dev

  echo "   Attente du démarrage (cela peut prendre plusieurs minutes)..."
  aws rds wait db-instance-available --db-instance-identifier harena-db-dev

  echo "   ✓ RDS démarré"
elif [ "$RDS_STATUS" = "available" ]; then
  echo "   ✓ RDS déjà actif"
else
  echo "   RDS status: $RDS_STATUS"
fi

RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "   RDS Endpoint: $RDS_ENDPOINT"
echo ""

# Vérifier Redis
echo "3. Vérification d'ElastiCache Redis..."
REDIS_STATUS=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id harena-redis-dev \
  --query 'CacheClusters[0].CacheClusterStatus' \
  --output text 2>/dev/null || echo "not-found")

if [ "$REDIS_STATUS" = "not-found" ]; then
  echo "   ⚠ ElastiCache n'existe plus"
else
  echo "   ✓ ElastiCache status: $REDIS_STATUS"
fi

REDIS_ENDPOINT=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id harena-redis-dev \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text 2>/dev/null || echo "")

if [ -n "$REDIS_ENDPOINT" ]; then
  echo "   Redis Endpoint: $REDIS_ENDPOINT"
fi
echo ""

# Restaurer PostgreSQL dans RDS
echo "4. Restauration de PostgreSQL dans RDS..."

# Lire les credentials
DB_PASSWORD=${DB_PASSWORD:-$(grep DB_PASSWORD .env | cut -d '=' -f2)}
DB_USERNAME=${DB_USERNAME:-harena_admin}
DB_NAME=${DB_NAME:-harena}

echo "   Restauration de la base de données..."
PGPASSWORD=$DB_PASSWORD pg_restore \
  -h $RDS_ENDPOINT \
  -U $DB_USERNAME \
  -d $DB_NAME \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  --verbose \
  "$BACKUP_DIR/postgres_backup.dump" || true

echo "   ✓ PostgreSQL restauré dans RDS"
echo ""

# Restaurer Elasticsearch sur l'ancienne instance
echo "5. Restauration d'Elasticsearch sur l'ancienne instance..."

# Upload du backup
if [ -f "$BACKUP_DIR/elasticsearch_backup.tar.gz" ]; then
  scp -o StrictHostKeyChecking=no "$BACKUP_DIR/elasticsearch_backup.tar.gz" ec2-user@$OLD_INSTANCE_IP:/tmp/

  ssh ec2-user@$OLD_INSTANCE_IP "
    set -e

    # Arrêter Elasticsearch
    sudo systemctl stop elasticsearch || true

    # Restaurer les données
    cd /tmp
    tar xzf elasticsearch_backup.tar.gz
    sudo rm -rf /var/lib/elasticsearch/*
    sudo cp -r elasticsearch_backup/* /var/lib/elasticsearch/
    sudo chown -R elasticsearch:elasticsearch /var/lib/elasticsearch

    # Redémarrer Elasticsearch
    sudo systemctl start elasticsearch

    echo 'Attente du démarrage d'\''Elasticsearch...'
    for i in {1..60}; do
      if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo '✓ Elasticsearch démarré'
        break
      fi
      sleep 2
    done
  "

  echo "   ✓ Elasticsearch restauré"
fi

echo ""

# Redémarrer le backend sur l'ancienne instance
echo "6. Redémarrage du backend sur l'ancienne instance..."
ssh ec2-user@$OLD_INSTANCE_IP "
  set -e
  cd /home/ec2-user/harena

  # Redémarrer le service
  sudo systemctl restart conversation-service

  echo 'Attente du démarrage...'
  sleep 10

  # Vérifier
  if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo '✓ Backend démarré'
  else
    echo '⚠ Backend ne répond pas encore'
  fi
"

echo "   ✓ Backend redémarré"
echo ""

# Arrêter la nouvelle infrastructure
echo "7. Arrêt de la nouvelle infrastructure (optionnel)..."
read -p "Voulez-vous arrêter la nouvelle infrastructure? (yes/no): " STOP_NEW

if [ "$STOP_NEW" = "yes" ]; then
  NEW_INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=harena-allinone-dev" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

  if [ "$NEW_INSTANCE_ID" != "None" ] && [ -n "$NEW_INSTANCE_ID" ]; then
    echo "   Arrêt de la nouvelle instance: $NEW_INSTANCE_ID"
    aws ec2 stop-instances --instance-ids $NEW_INSTANCE_ID
    echo "   ✓ Nouvelle instance arrêtée"
  fi
fi

echo ""
echo "=========================================="
echo "ROLLBACK TERMINÉ"
echo "=========================================="
echo ""
echo "Ancienne infrastructure restaurée:"
echo "  - Instance EC2: $OLD_INSTANCE_IP"
echo "  - RDS: $RDS_ENDPOINT"
echo "  - Redis: $REDIS_ENDPOINT"
echo ""
echo "Vérifications à faire:"
echo "  1. Tester l'API: http://$OLD_INSTANCE_IP:8000"
echo "  2. Vérifier les logs: ssh ec2-user@$OLD_INSTANCE_IP 'sudo journalctl -u conversation-service -f'"
echo "  3. Vérifier les données dans PostgreSQL"
echo "  4. Vérifier les données dans Elasticsearch"
echo ""
echo "Si tout fonctionne, vous pouvez supprimer la nouvelle infrastructure:"
echo "  cd terraform && terraform destroy -target=module.ec2_allinone"
echo ""
