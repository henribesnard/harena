#!/bin/bash
# Script de lancement local du budget profiling service

# Se placer à la racine du projet
cd "$(dirname "$0")/.."

# Définir PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Lancer le service
python -m budget_profiling_service.main
