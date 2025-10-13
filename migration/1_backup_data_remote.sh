#!/bin/bash
set -e

# Script de backup à exécuter DEPUIS l'instance EC2
# Ce script est uploadé et exécuté sur l'instance EC2 qui a accès à RDS

BACKUP_DIR="/tmp/harena_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "BACKUP DEPUIS EC2 - $(date)"
echo "=========================================="
echo "Dossier: $BACKUP_DIR"
echo ""

# Récupérer les variables d'environnement
cd /home/ec2-user/harena/conversation_service
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# Récupérer le RDS endpoint depuis AWS
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "RDS Endpoint: $RDS_ENDPOINT"
echo ""

# 1. Backup PostgreSQL depuis RDS
echo "1. Backup PostgreSQL..."
PGPASSWORD=$DB_PASSWORD pg_dump \
  -h $RDS_ENDPOINT \
  -U ${DB_USERNAME:-harena_admin} \
  -d ${DB_NAME:-harena} \
  --verbose \
  --format=custom \
  -f "$BACKUP_DIR/postgres_backup.dump"

echo "   ✓ PostgreSQL: $(du -h $BACKUP_DIR/postgres_backup.dump | cut -f1)"
echo ""

# 2. Backup Elasticsearch
echo "2. Backup Elasticsearch..."
if curl -s http://localhost:9200/_cluster/health > /dev/null; then
  # Créer un snapshot
  curl -X PUT 'http://localhost:9200/_snapshot/backup_repo' -H 'Content-Type: application/json' -d '{
    "type": "fs",
    "settings": {
      "location": "/tmp/es_snapshot"
    }
  }' 2>/dev/null || true

  mkdir -p /tmp/es_snapshot
  sudo chown -R elasticsearch:elasticsearch /tmp/es_snapshot || true

  SNAPSHOT_NAME="migration_$(date +%Y%m%d_%H%M%S)"
  curl -X PUT "http://localhost:9200/_snapshot/backup_repo/$SNAPSHOT_NAME?wait_for_completion=true" \
    -H 'Content-Type: application/json' -d '{
      "indices": "*",
      "ignore_unavailable": true,
      "include_global_state": false
    }' 2>/dev/null

  # Archive du snapshot
  sudo tar czf "$BACKUP_DIR/elasticsearch_backup.tar.gz" -C /tmp es_snapshot/

  echo "   ✓ Elasticsearch snapshot: $(du -h $BACKUP_DIR/elasticsearch_backup.tar.gz | cut -f1)"

  # Export JSON des indices (fallback)
  curl -s 'http://localhost:9200/_cat/indices?format=json' | \
    jq -r '.[].index | select(startswith(".") | not)' | \
    while read index; do
      echo "   Export index: $index"
      curl -s "http://localhost:9200/$index/_search?size=10000" > "$BACKUP_DIR/${index}_export.json"
    done

  tar czf "$BACKUP_DIR/elasticsearch_json_export.tar.gz" -C "$BACKUP_DIR" *_export.json
  rm -f "$BACKUP_DIR"/*_export.json

  echo "   ✓ Elasticsearch JSON export: $(du -h $BACKUP_DIR/elasticsearch_json_export.tar.gz | cut -f1)"
else
  echo "   ⚠ Elasticsearch non accessible"
fi

echo ""

# 3. Backup configuration
echo "3. Backup configuration..."
cp /home/ec2-user/harena/conversation_service/.env "$BACKUP_DIR/env_backup" || true
sudo tar czf "$BACKUP_DIR/systemd_services.tar.gz" \
  /etc/systemd/system/conversation-service.service \
  /etc/systemd/system/elasticsearch.service 2>/dev/null || true

echo "   ✓ Configuration sauvegardée"
echo ""

# 4. Créer une archive globale
echo "4. Création de l'archive finale..."
cd /tmp
ARCHIVE_NAME="harena_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$ARCHIVE_NAME" "$(basename $BACKUP_DIR)"

echo "   ✓ Archive créée: $ARCHIVE_NAME ($(du -h /tmp/$ARCHIVE_NAME | cut -f1))"
echo ""

echo "=========================================="
echo "BACKUP TERMINÉ"
echo "=========================================="
echo "Archive: /tmp/$ARCHIVE_NAME"
echo ""
echo "Pour télécharger:"
echo "  scp ec2-user@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):/tmp/$ARCHIVE_NAME ."
