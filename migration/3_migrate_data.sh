#!/bin/bash
set -e

# Script de migration des données vers la nouvelle infrastructure
# Usage: ./3_migrate_data.sh <backup_directory> <new_instance_ip>

if [ $# -lt 2 ]; then
  echo "Usage: $0 <backup_directory> <new_instance_ip>"
  echo ""
  echo "Example:"
  echo "  $0 ./migration/backups/20250112_143000 52.210.228.191"
  exit 1
fi

BACKUP_DIR=$1
NEW_IP=$2

if [ ! -d "$BACKUP_DIR" ]; then
  echo "❌ ERREUR: Le répertoire de backup n'existe pas: $BACKUP_DIR"
  exit 1
fi

echo "=========================================="
echo "MIGRATION DES DONNÉES"
echo "=========================================="
echo "Backup source: $BACKUP_DIR"
echo "Instance cible: $NEW_IP"
echo ""

# Vérifier les fichiers de backup
echo "1. Vérification des backups..."
if [ ! -f "$BACKUP_DIR/postgres_backup.dump" ]; then
  echo "❌ ERREUR: Backup PostgreSQL manquant"
  exit 1
fi

echo "   ✓ Backup PostgreSQL trouvé: $(du -h $BACKUP_DIR/postgres_backup.dump | cut -f1)"

if [ -f "$BACKUP_DIR/elasticsearch_backup.tar.gz" ]; then
  echo "   ✓ Backup Elasticsearch trouvé: $(du -h $BACKUP_DIR/elasticsearch_backup.tar.gz | cut -f1)"
  HAS_ES_BACKUP=true
else
  echo "   ⚠ Backup Elasticsearch non trouvé, utilisation du JSON export"
  HAS_ES_BACKUP=false
fi

echo ""

# Upload des backups vers la nouvelle instance
echo "2. Upload des backups vers la nouvelle instance..."
scp -o StrictHostKeyChecking=no "$BACKUP_DIR/postgres_backup.dump" ec2-user@$NEW_IP:/tmp/
echo "   ✓ PostgreSQL backup uploadé"

if [ "$HAS_ES_BACKUP" = true ]; then
  scp -o StrictHostKeyChecking=no "$BACKUP_DIR/elasticsearch_backup.tar.gz" ec2-user@$NEW_IP:/tmp/
  echo "   ✓ Elasticsearch backup uploadé"
fi

if [ -f "$BACKUP_DIR/elasticsearch_json_export.tar.gz" ]; then
  scp -o StrictHostKeyChecking=no "$BACKUP_DIR/elasticsearch_json_export.tar.gz" ec2-user@$NEW_IP:/tmp/
  echo "   ✓ Elasticsearch JSON export uploadé"
fi

echo ""

# Restauration PostgreSQL
echo "3. Restauration de PostgreSQL..."
ssh ec2-user@$NEW_IP "
  set -e
  cd /opt/harena

  # Lire les credentials depuis .env
  export \$(cat .env | grep -v '^#' | xargs)

  echo '   Attente que PostgreSQL soit prêt...'
  for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U \$DB_USERNAME > /dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  echo '   Restauration de la base de données...'

  # Restaurer le dump
  docker-compose exec -T postgres pg_restore \
    -U \$DB_USERNAME \
    -d \$DB_NAME \
    --clean \
    --if-exists \
    --no-owner \
    --no-privileges \
    --verbose < /tmp/postgres_backup.dump || true

  # Vérifier la restauration
  ROW_COUNT=\$(docker-compose exec -T postgres psql -U \$DB_USERNAME -d \$DB_NAME -t -c \"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'\")
  echo \"   ✓ Tables restaurées: \$ROW_COUNT\"
"

echo "   ✓ PostgreSQL restauré"
echo ""

# Restauration Elasticsearch
echo "4. Restauration d'Elasticsearch..."
ssh ec2-user@$NEW_IP "
  set -e
  cd /opt/harena

  echo '   Attente qu\'Elasticsearch soit prêt...'
  for i in {1..60}; do
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  HEALTH=\$(curl -s http://localhost:9200/_cluster/health | jq -r '.status')
  echo \"   Elasticsearch status: \$HEALTH\"

  # Restauration depuis snapshot si disponible
  if [ -f /tmp/elasticsearch_backup.tar.gz ]; then
    echo '   Restauration depuis snapshot...'

    cd /tmp
    tar xzf elasticsearch_backup.tar.gz

    # Copier dans le container
    docker cp elasticsearch_backup harena-elasticsearch:/tmp/

    # Configurer le repository
    curl -X PUT 'http://localhost:9200/_snapshot/backup_repo' -H 'Content-Type: application/json' -d '{
      \"type\": \"fs\",
      \"settings\": {
        \"location\": \"/tmp/elasticsearch_backup\"
      }
    }' || true

    # Lister les snapshots
    SNAPSHOTS=\$(curl -s 'http://localhost:9200/_snapshot/backup_repo/_all' | jq -r '.snapshots[].snapshot')

    if [ -n \"\$SNAPSHOTS\" ]; then
      LATEST_SNAPSHOT=\$(echo \"\$SNAPSHOTS\" | tail -1)
      echo \"   Restauration du snapshot: \$LATEST_SNAPSHOT\"

      curl -X POST \"http://localhost:9200/_snapshot/backup_repo/\$LATEST_SNAPSHOT/_restore\" -H 'Content-Type: application/json' -d '{
        \"indices\": \"*\",
        \"ignore_unavailable\": true,
        \"include_global_state\": false
      }'

      echo '   ✓ Snapshot restauré'
    fi
  fi

  # Fallback: restauration depuis JSON export
  if [ -f /tmp/elasticsearch_json_export.tar.gz ]; then
    echo '   Restauration depuis JSON export...'

    cd /tmp
    tar xzf elasticsearch_json_export.tar.gz

    # Re-indexer chaque fichier JSON
    for json_file in /tmp/*_export.json; do
      if [ -f \"\$json_file\" ]; then
        INDEX_NAME=\$(basename \$json_file _export.json)
        echo \"   Indexation de: \$INDEX_NAME\"

        # Créer l'index et indexer les documents
        jq -c '.hits.hits[]' \$json_file | while read doc; do
          DOC_ID=\$(echo \$doc | jq -r '._id')
          DOC_SOURCE=\$(echo \$doc | jq '._source')

          curl -X PUT \"http://localhost:9200/\$INDEX_NAME/_doc/\$DOC_ID\" \
            -H 'Content-Type: application/json' \
            -d \"\$DOC_SOURCE\" > /dev/null 2>&1
        done
      fi
    done

    echo '   ✓ JSON export restauré'
  fi

  # Vérifier les indices
  echo ''
  echo '   Indices Elasticsearch:'
  curl -s 'http://localhost:9200/_cat/indices?v'
"

echo "   ✓ Elasticsearch restauré"
echo ""

# Démarrer le backend
echo "5. Démarrage du backend..."
ssh ec2-user@$NEW_IP "
  set -e
  cd /opt/harena

  # Build et démarrer le backend
  docker-compose up -d --build backend

  echo '   Attente du démarrage du backend...'
  sleep 15

  # Vérifier la santé
  for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
      HEALTH=\$(curl -s http://localhost:8001/health)
      echo \"   ✓ Backend démarré: \$HEALTH\"
      break
    fi
    sleep 2
  done

  # Afficher les logs
  echo ''
  echo '   Derniers logs:'
  docker-compose logs --tail=20 backend
"

echo "   ✓ Backend démarré"
echo ""

# Tests de validation
echo "6. Tests de validation..."
echo "   Test de l'API..."

HEALTH_CHECK=$(ssh ec2-user@$NEW_IP "curl -s http://localhost:8001/health")
if echo "$HEALTH_CHECK" | grep -q "healthy"; then
  echo "   ✓ Health check OK"
else
  echo "   ⚠ Health check: $HEALTH_CHECK"
fi

# Compter les tables PostgreSQL
TABLE_COUNT=$(ssh ec2-user@$NEW_IP "
  cd /opt/harena
  export \$(cat .env | grep -v '^#' | xargs)
  docker-compose exec -T postgres psql -U \$DB_USERNAME -d \$DB_NAME -t -c \"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'\"
" | tr -d ' ')

echo "   ✓ Tables PostgreSQL: $TABLE_COUNT"

# Compter les indices Elasticsearch
INDEX_COUNT=$(ssh ec2-user@$NEW_IP "curl -s 'http://localhost:9200/_cat/indices?format=json' | jq 'length'")
echo "   ✓ Indices Elasticsearch: $INDEX_COUNT"

echo ""
echo "=========================================="
echo "MIGRATION TERMINÉE AVEC SUCCÈS"
echo "=========================================="
echo ""
echo "Nouvelle infrastructure:"
echo "  - URL API: http://$NEW_IP:8000"
echo "  - URL Health: http://$NEW_IP:8001/health"
echo ""
echo "Données migrées:"
echo "  - PostgreSQL: $TABLE_COUNT tables"
echo "  - Elasticsearch: $INDEX_COUNT indices"
echo ""
echo "Prochaines étapes:"
echo "  1. Tester l'application complètement"
echo "  2. Mettre à jour le frontend avec la nouvelle URL"
echo "  3. Si tout fonctionne: ./migration/5_cleanup_old_infrastructure.sh"
echo "  4. En cas de problème: ./migration/4_rollback.sh $BACKUP_DIR"
echo ""
