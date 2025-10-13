#!/bin/sh
set -e

# Script de backup automatique journalier
# Lance des backups PostgreSQL et Elasticsearch tous les jours à 2h du matin

echo "Installing required packages..."
apk add --no-cache postgresql-client curl aws-cli

# Créer le script de backup
cat > /usr/local/bin/daily_backup.sh <<'EOF'
#!/bin/sh
set -e

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/daily/$BACKUP_DATE"
mkdir -p "$BACKUP_DIR"

echo "=== Starting daily backup at $(date) ==="

# Backup PostgreSQL
echo "Backing up PostgreSQL..."
PGPASSWORD=$POSTGRES_PASSWORD pg_dump \
  -h postgres \
  -U $POSTGRES_USER \
  -d $POSTGRES_DB \
  --format=custom \
  -f "$BACKUP_DIR/postgres_backup.dump"

echo "PostgreSQL backup completed: $(du -h $BACKUP_DIR/postgres_backup.dump | cut -f1)"

# Backup Elasticsearch
echo "Backing up Elasticsearch..."
# Utiliser l'API snapshot si configuré, sinon copier les données
if curl -s http://elasticsearch:9200/_cluster/health > /dev/null; then
  # Créer un export JSON des indices
  curl -s 'http://elasticsearch:9200/_cat/indices?format=json' | \
    jq -r '.[].index | select(startswith(".") | not)' | \
    while read index; do
      curl -s "http://elasticsearch:9200/$index/_search?size=10000" > "$BACKUP_DIR/${index}_export.json"
    done
  echo "Elasticsearch JSON export completed"
fi

# Compression
cd "$BACKUP_DIR"
tar czf "../backup_$BACKUP_DATE.tar.gz" .
cd ..
rm -rf "$BACKUP_DATE"

echo "Backup completed: $(du -h backup_$BACKUP_DATE.tar.gz | cut -f1)"

# Upload vers S3 si configuré
if [ -n "$S3_BUCKET" ]; then
  echo "Uploading to S3..."
  aws s3 cp "backup_$BACKUP_DATE.tar.gz" "s3://$S3_BUCKET/backups/"
  echo "Upload completed"
fi

# Nettoyage des anciens backups
echo "Cleaning up old backups (retention: $BACKUP_RETENTION_DAYS days)..."
find /backups/daily -name "backup_*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete
echo "Cleanup completed"

echo "=== Backup finished at $(date) ==="
EOF

chmod +x /usr/local/bin/daily_backup.sh

# Créer le crontab
echo "0 2 * * * /usr/local/bin/daily_backup.sh >> /backups/backup.log 2>&1" | crontab -

echo "Backup cron job configured. Running crond..."
crond -f -l 2
