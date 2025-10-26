#!/bin/bash

# ============================================
# HARENA - SCRIPT DE BACKUP AWS
# ============================================
# Backup automatique de la base de données
# ============================================

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
EC2_IP="${EC2_IP:-63.35.52.216}"
EC2_USER="${EC2_USER:-ec2-user}"
SSH_KEY="${SSH_KEY:-~/.ssh/harena-aws.pem}"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}HARENA - BACKUP BASE DE DONNÉES${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Fonction pour exécuter des commandes SSH
ssh_exec() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" "$@"
}

# Nom du fichier backup
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="harena_backup_$DATE.sql.gz"

echo -e "${YELLOW}Création du backup...${NC}"
ssh_exec << EOF
    cd ~/harena

    # Créer le dossier backups s'il n'existe pas
    mkdir -p ~/harena/backups

    # Dump de la base de données
    docker exec harena_postgres pg_dump -U harena_admin harena | gzip > ~/harena/backups/$BACKUP_FILE

    # Afficher la taille
    du -h ~/harena/backups/$BACKUP_FILE

    # Nettoyer les anciens backups (garder seulement les 7 derniers)
    cd ~/harena/backups
    ls -t harena_backup_*.sql.gz | tail -n +8 | xargs -r rm

    echo ""
    echo "Backups disponibles:"
    ls -lh harena_backup_*.sql.gz
EOF

echo ""
echo -e "${GREEN}✓ Backup créé: $BACKUP_FILE${NC}"
echo ""
echo "📥 Télécharger le backup:"
echo "  scp -i $SSH_KEY $EC2_USER@$EC2_IP:~/harena/backups/$BACKUP_FILE ."
echo ""
echo "♻️  Restaurer un backup:"
echo "  gunzip < $BACKUP_FILE | ssh -i $SSH_KEY $EC2_USER@$EC2_IP 'docker exec -i harena_postgres psql -U harena_admin harena'"
