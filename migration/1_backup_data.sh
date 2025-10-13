#!/bin/bash
set -e

# Script de backup complet avant migration
# Ce script sauvegarde PostgreSQL et Elasticsearch

BACKUP_DIR="./migration/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "BACKUP AVANT MIGRATION - $(date)"
echo "=========================================="
echo "Dossier de backup: $BACKUP_DIR"
echo ""

# Récupérer les endpoints depuis AWS
echo "1. Récupération des endpoints AWS..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=harena-backend-*" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "Instance EC2: $INSTANCE_ID ($PUBLIC_IP)"
echo "RDS Endpoint: $RDS_ENDPOINT"
echo ""

# Backup PostgreSQL depuis RDS
echo "2. Backup PostgreSQL depuis RDS..."
echo "   Connexion à $RDS_ENDPOINT..."

# Lire les credentials depuis les variables d'environnement ou .env
DB_PASSWORD=${DB_PASSWORD:-$(grep DB_PASSWORD .env | cut -d '=' -f2)}
DB_USERNAME=${DB_USERNAME:-harena_admin}
DB_NAME=${DB_NAME:-harena}

# Backup complet de la base
PGPASSWORD=$DB_PASSWORD pg_dump \
  -h $RDS_ENDPOINT \
  -U $DB_USERNAME \
  -d $DB_NAME \
  --verbose \
  --format=custom \
  -f "$BACKUP_DIR/postgres_backup.dump"

echo "   ✓ PostgreSQL backup créé: $(du -h $BACKUP_DIR/postgres_backup.dump | cut -f1)"
echo ""

# Backup Elasticsearch depuis EC2
echo "3. Backup Elasticsearch depuis EC2..."
echo "   Connexion à l'instance EC2..."

# Créer un snapshot Elasticsearch via l'API
ssh -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP "
  set -e

  # Vérifier si Elasticsearch est actif
  if ! curl -s http://localhost:9200/_cluster/health > /dev/null; then
    echo '   ⚠ Elasticsearch non accessible, skip backup'
    exit 0
  fi

  echo '   Création du snapshot Elasticsearch...'

  # Configurer le repository de snapshot
  curl -X PUT 'http://localhost:9200/_snapshot/backup_repo' -H 'Content-Type: application/json' -d '{
    \"type\": \"fs\",
    \"settings\": {
      \"location\": \"/tmp/elasticsearch_backup\"
    }
  }' || true

  # Créer le snapshot
  SNAPSHOT_NAME=\"migration_$(date +%Y%m%d_%H%M%S)\"
  curl -X PUT \"http://localhost:9200/_snapshot/backup_repo/\$SNAPSHOT_NAME?wait_for_completion=true\" -H 'Content-Type: application/json' -d '{
    \"indices\": \"*\",
    \"ignore_unavailable\": true,
    \"include_global_state\": false
  }'

  # Créer une archive du snapshot
  cd /tmp
  sudo tar czf elasticsearch_backup.tar.gz elasticsearch_backup/
  echo '   ✓ Snapshot Elasticsearch créé'
"

# Télécharger le backup Elasticsearch
echo "   Téléchargement du backup Elasticsearch..."
scp -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP:/tmp/elasticsearch_backup.tar.gz "$BACKUP_DIR/"

echo "   ✓ Elasticsearch backup téléchargé: $(du -h $BACKUP_DIR/elasticsearch_backup.tar.gz | cut -f1)"
echo ""

# Backup des indices Elasticsearch en JSON (fallback)
echo "4. Export des indices Elasticsearch en JSON..."
ssh ec2-user@$PUBLIC_IP "
  if curl -s http://localhost:9200/_cluster/health > /dev/null; then
    # Lister tous les indices
    curl -s 'http://localhost:9200/_cat/indices?format=json' | \
      jq -r '.[].index | select(startswith(\".\") | not)' | \
      while read index; do
        echo \"   Export de l'index: \$index\"
        curl -s \"http://localhost:9200/\$index/_search?size=10000\" > \"/tmp/\${index}_export.json\"
      done

    tar czf /tmp/elasticsearch_json_export.tar.gz /tmp/*_export.json
    echo '   ✓ Export JSON créé'
  fi
" || echo "   ⚠ Skip JSON export"

scp -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP:/tmp/elasticsearch_json_export.tar.gz "$BACKUP_DIR/" || echo "   ⚠ Pas de JSON export disponible"
echo ""

# Backup de la configuration actuelle
echo "5. Backup de la configuration..."
ssh ec2-user@$PUBLIC_IP "
  cd /home/ec2-user/harena

  # Backup du .env
  sudo cp conversation_service/.env /tmp/harena_env_backup
  sudo chmod 644 /tmp/harena_env_backup

  # Backup des services systemd
  sudo tar czf /tmp/systemd_services.tar.gz /etc/systemd/system/conversation-service.service /etc/systemd/system/elasticsearch.service || true
"

scp -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP:/tmp/harena_env_backup "$BACKUP_DIR/.env.backup"
scp -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP:/tmp/systemd_services.tar.gz "$BACKUP_DIR/" || true

echo "   ✓ Configuration sauvegardée"
echo ""

# Créer un fichier de métadonnées
echo "6. Création du fichier de métadonnées..."
cat > "$BACKUP_DIR/backup_metadata.json" <<EOF
{
  "backup_date": "$(date -Iseconds)",
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP",
  "rds_endpoint": "$RDS_ENDPOINT",
  "db_name": "$DB_NAME",
  "db_username": "$DB_USERNAME",
  "files": [
    "postgres_backup.dump",
    "elasticsearch_backup.tar.gz",
    "elasticsearch_json_export.tar.gz",
    ".env.backup",
    "systemd_services.tar.gz"
  ]
}
EOF

echo "   ✓ Métadonnées créées"
echo ""

# Résumé
echo "=========================================="
echo "BACKUP TERMINÉ AVEC SUCCÈS"
echo "=========================================="
echo "Dossier: $BACKUP_DIR"
echo ""
echo "Fichiers créés:"
ls -lh "$BACKUP_DIR/"
echo ""
echo "Taille totale: $(du -sh $BACKUP_DIR | cut -f1)"
echo ""
echo "⚠ IMPORTANT: Gardez ce backup jusqu'à ce que la migration soit validée!"
echo "   Pour restaurer en cas de problème: ./migration/4_rollback.sh $BACKUP_DIR"
echo ""
