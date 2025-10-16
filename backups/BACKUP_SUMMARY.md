# Backup Harena - Pré-déploiement AWS
**Date:** 2025-10-16 22:20
**Version:** v4.0.1

## ✅ Backups PostgreSQL

**Serveur:** 63.35.52.216:5432
**Base:** harena
**Tables sauvegardées:** 18/18

### Données critiques:
- **raw_transactions:** 6,726 lignes
- **conversations:** 171 lignes
- **conversation_turns:** 171 lignes
- **categories:** 37 lignes
- **category_groups:** 8 lignes
- **users:** 1 ligne
- **sync_accounts:** 1 ligne
- **sync_items:** 1 ligne

### Fichiers:
```
backups/alembic_version_20251016_221550.csv
backups/raw_transactions_20251016_221550.csv
backups/conversations_20251016_221550.csv
backups/categories_20251016_221550.csv
... (18 fichiers CSV)
backups/backup_metadata_20251016_221550.txt
```

## ✅ Backups Elasticsearch

**Serveur:** Bonsai (fir-178893546.eu-west-1.bonsaisearch.net:443)
**Indices:**
- harena_transactions-000001: 13,454 documents
- harena_accounts: 2 documents

### Fichiers:
```
backups/elasticsearch_harena_transactions_YYYYMMDD_HHMMSS.json (9.4 MB)
backups/elasticsearch_mappings_20251016_222027.json
backups/elasticsearch_settings_20251016_222027.json
backups/elasticsearch_accounts_20251016_222027.json
```

## 📋 Configuration Locale Actuelle

### Services Backend (tous sur port 8000 en local):
- **user_service:** Port 8000 (géré directement)
- **search_service:** Port 8000 (via local_app.py)
- **conversation_service:** Port 8001 (actuellement, **à changer pour 8000 sur AWS**)

### Frontend:
- **Répertoire:** `harena_front/`
- **Configuration:** `.env` avec `VITE_API_URL`
- **Fichiers à modifier pour AWS:**
  - `harena_front/.env`
  - `harena_front/src/services/api.ts`
  - `harena_front/src/services/api/coreMetricsApi.ts`
  - `harena_front/src/services/api/metricsApi.ts`

**URL actuelle:** `http://localhost:8000`
**URL AWS à configurer:** À définir après déploiement

## 🗄️ Base de Données

### PostgreSQL:
- **Total:** 6,726 transactions
- **Utilisateurs:** 1
- **Catégories:** 37
- **Conversations:** 171

### Elasticsearch:
- **Transactions indexées:** 13,454
- **Comptes:** 2

## 📦 Dépendances Externes

- **Bonsai Elasticsearch:** Hébergé (https://fir-178893546.eu-west-1.bonsaisearch.net:443)
- **PostgreSQL:** Hébergé (63.35.52.216:5432)

## ⚠️ Points d'Attention pour AWS

1. **Ports uniformes:** Tous les services backend sur port 8000
2. **conversation_service:** Actuellement sur 8001, à aligner sur 8000
3. **Frontend:** Mettre à jour VITE_API_URL avec l'URL AWS finale
4. **Elasticsearch:** Bonsai restera hébergé externellement
5. **PostgreSQL:** Base actuelle sur 63.35.52.216 restera externelle (ou migration vers RDS)

## 🔐 Secrets à Configurer

- `DATABASE_URL`
- `BONSAI_URL`
- `SECRET_KEY` (JWT)
- `DEEPSEEK_API_KEY`

## 📝 Notes

- Backup complet réalisé avant déploiement AWS
- Version v4.0.1 incluant fix classification "retraits espèces"
- Tous les backups stockés dans `backups/`
- Prêt pour déploiement infrastructure AWS
