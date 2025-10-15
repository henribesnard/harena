#!/bin/bash
# Script complet pour mettre à jour PostgreSQL et Elasticsearch
# À exécuter sur le serveur EC2

set -e  # Arrêter en cas d'erreur

echo "========================================================================"
echo "MISE À JOUR COMPLÈTE POSTGRESQL + ELASTICSEARCH"
echo "========================================================================"
echo ""

cd /home/ec2-user/harena

# Étape 1: Mise à jour PostgreSQL
echo "========================================================================"
echo "ÉTAPE 1/2 : MISE À JOUR POSTGRESQL"
echo "========================================================================"
echo ""
echo "[=>] Mise à jour des champs:"
echo "    - operation_type (depuis CSV)"
echo "    - amount (depuis CSV)"
echo "    - clean_description (depuis CSV)"
echo "    - category_name (depuis CSV)"
echo "    - merchant_name (depuis CSV)"
echo "    - booking_date, transaction_date, value_date (copie depuis date)"
echo ""

if [ ! -f "analyse_specifique/export_postgres_transactions_user_100_2.csv" ]; then
    echo "[ERROR] Fichier CSV introuvable: analyse_specifique/export_postgres_transactions_user_100_2.csv"
    exit 1
fi

python3 analyse_specifique/06_update_operation_type.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Échec de la mise à jour PostgreSQL"
    exit 1
fi

echo ""
echo "[OK] PostgreSQL mis à jour avec succès"

# Pause entre les deux étapes
sleep 2

# Étape 2: Synchronisation vers Elasticsearch
echo ""
echo "========================================================================"
echo "ÉTAPE 2/2 : SYNCHRONISATION ELASTICSEARCH"
echo "========================================================================"
echo ""
echo "[=>] Synchronisation des données PostgreSQL vers Elasticsearch"
echo "    Les champs mis à jour seront automatiquement synchronisés:"
echo "    - operation_type"
echo "    - amount"
echo "    - merchant_name"
echo "    - category_name"
echo "    - booking_date, transaction_date, value_date"
echo ""

python3 analyse_specifique/05_manual_enrichment_elasticsearch.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Échec de la synchronisation Elasticsearch"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[OK] MISE À JOUR COMPLÈTE TERMINÉE AVEC SUCCÈS"
echo "========================================================================"
echo ""
echo "Résumé:"
echo "  1. PostgreSQL: operation_type, amount, dates mis à jour"
echo "  2. Elasticsearch: données synchronisées depuis PostgreSQL"
echo ""
echo "Les questions F015, F016, F017 peuvent maintenant utiliser operation_type"
echo ""
