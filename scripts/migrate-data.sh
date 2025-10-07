#!/bin/bash
set -e

echo "🔄 Migration des données Heroku → AWS RDS"
echo "=========================================="

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Fichier .env introuvable"
    exit 1
fi

# Get RDS endpoint from Terraform
cd terraform
RDS_ENDPOINT=$(terraform output -raw rds_endpoint 2>/dev/null || echo "")
cd ..

if [ -z "$RDS_ENDPOINT" ]; then
    echo "❌ RDS endpoint introuvable. Lancez d'abord 'terraform apply'"
    exit 1
fi

# Extract RDS host (remove port)
RDS_HOST="${RDS_ENDPOINT%:*}"

echo "📊 Source: Heroku PostgreSQL"
echo "   $POSTGRES_SERVER:$POSTGRES_PORT/$POSTGRES_DB"
echo ""
echo "🎯 Destination: AWS RDS"
echo "   $RDS_HOST:5432/harena"
echo ""

# Confirm migration
read -p "⚠️  Cette opération va écraser les données dans AWS RDS. Continuer? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Migration annulée"
    exit 0
fi

# Create backup of Heroku database
echo "💾 Sauvegarde de la base Heroku..."
BACKUP_FILE="/tmp/harena_backup_$(date +%Y%m%d_%H%M%S).sql"

PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
    -h "$POSTGRES_SERVER" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -p "$POSTGRES_PORT" \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    -f "$BACKUP_FILE"

echo "✅ Backup créé: $BACKUP_FILE"
echo "📏 Taille: $(du -h $BACKUP_FILE | cut -f1)"

# Test AWS RDS connection
echo "🔌 Test de connexion AWS RDS..."
PGPASSWORD="$AWS_DB_PASSWORD" psql \
    -h "$RDS_HOST" \
    -U "harena_admin" \
    -d "harena" \
    -p 5432 \
    -c "SELECT version();" > /dev/null

echo "✅ Connexion AWS RDS OK"

# Restore to AWS RDS
echo "📥 Restauration vers AWS RDS..."
PGPASSWORD="$AWS_DB_PASSWORD" psql \
    -h "$RDS_HOST" \
    -U "harena_admin" \
    -d "harena" \
    -p 5432 \
    -f "$BACKUP_FILE"

echo "✅ Données migrées avec succès !"

# Verify data
echo "🔍 Vérification des données..."

# Count tables
HEROKU_TABLES=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$POSTGRES_SERVER" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -p "$POSTGRES_PORT" \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)

AWS_TABLES=$(PGPASSWORD="$AWS_DB_PASSWORD" psql \
    -h "$RDS_HOST" \
    -U "harena_admin" \
    -d "harena" \
    -p 5432 \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)

echo "   Tables Heroku: $HEROKU_TABLES"
echo "   Tables AWS: $AWS_TABLES"

if [ "$HEROKU_TABLES" = "$AWS_TABLES" ]; then
    echo "✅ Nombre de tables identique"
else
    echo "⚠️  Différence dans le nombre de tables"
fi

# Keep backup
echo ""
echo "💾 Backup conservé dans: $BACKUP_FILE"
echo "   (Supprimez-le manuellement quand vous êtes sûr de la migration)"
echo ""
echo "✅ Migration terminée !"
echo ""
echo "📝 Prochaines étapes:"
echo "  1. Vérifiez que l'application fonctionne sur AWS"
echo "  2. Testez toutes les fonctionnalités critiques"
echo "  3. Si tout est OK, vous pouvez arrêter Heroku"
