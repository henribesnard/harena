#!/bin/bash
# Script à exécuter sur le serveur EC2 pour mettre à jour PostgreSQL

echo "========================================"
echo "MISE À JOUR POSTGRESQL DEPUIS CSV"
echo "========================================"

cd /home/ec2-user/harena

# Vérifier que le CSV existe
if [ ! -f "analyse_specifique/export_postgres_transactions_user_100_2.csv" ]; then
    echo "[ERROR] Fichier CSV introuvable"
    exit 1
fi

# Exécuter le script Python
python3 analyse_specifique/06_update_operation_type.py

echo ""
echo "========================================"
echo "MISE À JOUR TERMINÉE"
echo "========================================"
