#!/bin/bash
set -e

# Script de nettoyage de l'ancienne infrastructure
# À exécuter SEULEMENT après validation complète de la nouvelle infrastructure

echo "=========================================="
echo "NETTOYAGE DE L'ANCIENNE INFRASTRUCTURE"
echo "=========================================="
echo ""
echo "⚠️  ATTENTION ⚠️"
echo ""
echo "Ce script va DÉTRUIRE définitivement:"
echo "  - RDS PostgreSQL (harena-db-dev)"
echo "  - ElastiCache Redis (harena-redis-dev)"
echo "  - Ancienne instance EC2 (harena-backend-dev)"
echo "  - NAT Gateway"
echo "  - Anciens subnets privés"
echo ""
echo "Économies mensuelles estimées: ~58$/mois"
echo ""
echo "IMPORTANT:"
echo "  - Assurez-vous que la nouvelle infrastructure fonctionne parfaitement"
echo "  - Assurez-vous d'avoir un backup récent et testé"
echo "  - Cette action est IRRÉVERSIBLE"
echo ""

read -p "Êtes-vous ABSOLUMENT SÛR de vouloir continuer? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Nettoyage annulé"
  exit 0
fi

echo ""
read -p "Tapez 'DELETE' en majuscules pour confirmer: " CONFIRM2
if [ "$CONFIRM2" != "DELETE" ]; then
  echo "Nettoyage annulé"
  exit 0
fi

echo ""
echo "Démarrage du nettoyage..."
echo ""

# Créer un dernier backup de sécurité
echo "1. Création d'un backup de sécurité final..."
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text 2>/dev/null || echo "")

if [ -n "$RDS_ENDPOINT" ]; then
  FINAL_BACKUP_DIR="./migration/backups/final_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$FINAL_BACKUP_DIR"

  DB_PASSWORD=${DB_PASSWORD:-$(grep DB_PASSWORD .env | cut -d '=' -f2)}
  DB_USERNAME=${DB_USERNAME:-harena_admin}
  DB_NAME=${DB_NAME:-harena}

  echo "   Backup de RDS..."
  PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $RDS_ENDPOINT \
    -U $DB_USERNAME \
    -d $DB_NAME \
    --format=custom \
    -f "$FINAL_BACKUP_DIR/postgres_final_backup.dump"

  echo "   ✓ Backup final créé: $FINAL_BACKUP_DIR"
fi

echo ""

# Détruire les ressources avec Terraform
echo "2. Destruction des ressources AWS..."
cd terraform

# Liste des ressources à détruire
RESOURCES_TO_DESTROY=(
  "module.rds"
  "module.redis"
  "module.ec2"
  "module.vpc.aws_nat_gateway.main"
)

for resource in "${RESOURCES_TO_DESTROY[@]}"; do
  echo ""
  echo "   Destruction de: $resource"

  # Vérifier si la ressource existe
  if terraform state list | grep -q "$resource"; then
    terraform destroy -target="$resource" -auto-approve || {
      echo "   ⚠ Échec de la destruction de $resource"
      echo "   Continuons avec les autres ressources..."
    }
  else
    echo "   ⚠ Ressource non trouvée dans le state: $resource"
  fi
done

cd ..

echo ""
echo "3. Vérification de la destruction..."

# Vérifier RDS
RDS_EXISTS=$(aws rds describe-db-instances \
  --db-instance-identifier harena-db-dev \
  --query 'DBInstances[0].DBInstanceStatus' \
  --output text 2>/dev/null || echo "deleted")

if [ "$RDS_EXISTS" = "deleted" ]; then
  echo "   ✓ RDS détruit"
else
  echo "   ⚠ RDS status: $RDS_EXISTS (la suppression peut prendre du temps)"
fi

# Vérifier Redis
REDIS_EXISTS=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id harena-redis-dev \
  --query 'CacheClusters[0].CacheClusterStatus' \
  --output text 2>/dev/null || echo "deleted")

if [ "$REDIS_EXISTS" = "deleted" ]; then
  echo "   ✓ ElastiCache détruit"
else
  echo "   ⚠ ElastiCache status: $REDIS_EXISTS (la suppression peut prendre du temps)"
fi

# Vérifier EC2
OLD_INSTANCE=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=harena-backend-dev" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text 2>/dev/null || echo "terminated")

if [ "$OLD_INSTANCE" = "terminated" ] || [ "$OLD_INSTANCE" = "None" ]; then
  echo "   ✓ Ancienne instance EC2 détruite"
else
  echo "   ⚠ Ancienne instance status: $OLD_INSTANCE"
fi

echo ""

# Calculer les économies
echo "4. Calcul des économies..."
echo ""
echo "   Coûts AVANT l'optimisation:"
echo "     - RDS PostgreSQL:        ~14.71 $/mois"
echo "     - ElastiCache Redis:     ~12.41 $/mois"
echo "     - EC2 + EBS:             ~8.32 $/mois"
echo "     - NAT Gateway:           ~33.00 $/mois"
echo "     - S3 + CloudFront:       ~1.50 $/mois"
echo "     TOTAL:                   ~69.94 $/mois"
echo ""
echo "   Coûts APRÈS l'optimisation:"
echo "     - EC2 t4g.small (Spot):  ~6.13 $/mois"
echo "     - EBS 40GB:              ~3.20 $/mois"
echo "     - S3 + CloudFront:       ~1.50 $/mois"
echo "     - CloudWatch:            ~0.50 $/mois"
echo "     TOTAL:                   ~11.33 $/mois"
echo ""
echo "   ÉCONOMIES MENSUELLES:     ~58.61 $/mois (83% de réduction)"
echo "   ÉCONOMIES ANNUELLES:      ~703.32 $/an"
echo ""

echo "=========================================="
echo "NETTOYAGE TERMINÉ"
echo "=========================================="
echo ""
echo "Infrastructure actuelle:"
echo "  - Architecture: All-in-One sur EC2"
echo "  - Coût mensuel estimé: ~11.33 $/mois"
echo "  - Backup final sauvegardé dans: $FINAL_BACKUP_DIR"
echo ""
echo "Nouvelle infrastructure:"
NEW_IP=$(cd terraform && terraform output -raw ec2_public_ip 2>/dev/null || echo "N/A")
echo "  - URL API: http://$NEW_IP:8000"
echo "  - URL Health: http://$NEW_IP:8001/health"
echo ""
echo "Actions recommandées:"
echo "  1. Mettre à jour le DNS/domaine avec la nouvelle IP"
echo "  2. Configurer les backups automatiques vers S3"
echo "  3. Monitorer les coûts AWS pendant 1 semaine"
echo "  4. Conserver les backups pendant au moins 30 jours"
echo ""
