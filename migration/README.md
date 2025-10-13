# Migration vers l'architecture optimisée All-in-One

Ce dossier contient tous les scripts et fichiers nécessaires pour migrer de l'architecture actuelle (RDS + ElastiCache + NAT Gateway) vers une architecture optimisée tout-en-un sur une seule instance EC2.

## Objectif

Réduire les coûts AWS de **~70$/mois à ~11$/mois** (économie de 83%) tout en conservant toutes les fonctionnalités et les données.

## Architecture

### Avant (coût: ~70$/mois)
- RDS PostgreSQL db.t4g.micro: ~15$/mois
- ElastiCache Redis cache.t4g.micro: ~12$/mois
- EC2 t4g.micro + EBS: ~8$/mois
- NAT Gateway: ~33$/mois
- S3 + CloudFront: ~2$/mois

### Après (coût: ~11$/mois)
- EC2 t4g.small (Spot) avec Docker Compose: ~6$/mois
  - PostgreSQL (conteneur)
  - Redis (conteneur)
  - Elasticsearch (conteneur)
  - Backend API (conteneur)
- EBS 40GB: ~3$/mois
- S3 + CloudFront: ~2$/mois

## Processus de migration

### Étape 1: Backup des données actuelles
```bash
chmod +x migration/1_backup_data.sh
./migration/1_backup_data.sh
```

Ce script:
- Sauvegarde PostgreSQL depuis RDS (format dump)
- Sauvegarde Elasticsearch (snapshot + JSON export)
- Sauvegarde la configuration (.env, systemd services)
- Crée un dossier `backups/YYYYMMDD_HHMMSS/`

**IMPORTANT**: Ne passez PAS à l'étape suivante tant que ce backup n'est pas terminé et vérifié!

### Étape 2: Déploiement de la nouvelle infrastructure
```bash
chmod +x migration/2_deploy_new_infrastructure.sh
./migration/2_deploy_new_infrastructure.sh
```

Ce script:
- Déploie la nouvelle infrastructure Terraform optimisée
- Crée une instance EC2 t4g.small avec Docker pré-installé
- Upload le code de l'application
- Démarre les conteneurs PostgreSQL, Redis, Elasticsearch

**Durée estimée**: 10-15 minutes

### Étape 3: Migration des données
```bash
chmod +x migration/3_migrate_data.sh
./migration/3_migrate_data.sh <backup_dir> <new_instance_ip>
```

Exemple:
```bash
./migration/3_migrate_data.sh ./migration/backups/20250112_143000 52.210.228.191
```

Ce script:
- Upload les backups vers la nouvelle instance
- Restaure PostgreSQL
- Restaure Elasticsearch
- Démarre le backend
- Effectue des tests de validation

**Durée estimée**: 15-30 minutes (selon la taille des données)

### Étape 4: Tests et validation

Avant de continuer, testez TOUT:
- [ ] L'API répond correctement: `curl http://<new_ip>:8001/health`
- [ ] Les données PostgreSQL sont présentes
- [ ] Les indices Elasticsearch sont présents
- [ ] Les recherches fonctionnent
- [ ] L'authentification fonctionne
- [ ] Les analytics/visualizations fonctionnent
- [ ] Le cache Redis fonctionne

### Étape 5A: Cleanup (si tout fonctionne)
```bash
chmod +x migration/5_cleanup_old_infrastructure.sh
./migration/5_cleanup_old_infrastructure.sh
```

Ce script:
- Crée un backup final de sécurité
- Détruit RDS, ElastiCache, ancienne EC2, NAT Gateway
- Affiche les économies réalisées

**⚠️ ATTENTION**: Cette action est IRRÉVERSIBLE!

### Étape 5B: Rollback (si problème)
```bash
chmod +x migration/4_rollback.sh
./migration/4_rollback.sh <backup_dir>
```

Ce script:
- Redémarre l'ancienne infrastructure
- Restaure les données depuis le backup
- Arrête la nouvelle infrastructure (optionnel)

## Fichiers

### Scripts de migration
- `1_backup_data.sh` - Backup complet avant migration
- `2_deploy_new_infrastructure.sh` - Déploiement nouvelle infra
- `3_migrate_data.sh` - Migration des données
- `4_rollback.sh` - Rollback en cas de problème
- `5_cleanup_old_infrastructure.sh` - Nettoyage ancienne infra

### Configuration
- `docker-compose.yml` - Stack Docker complète
- `scripts/backup_cron.sh` - Backups automatiques journaliers

### Terraform
- `../terraform/main-optimized.tf` - Configuration Terraform optimisée
- `../terraform/modules/vpc-simple/` - VPC sans NAT Gateway
- `../terraform/modules/ec2-allinone/` - EC2 all-in-one
- `../terraform/modules/monitoring-simple/` - Monitoring simplifié

## Sécurité des données

### Backups automatiques
Les backups automatiques sont configurés via le conteneur `backup` dans docker-compose:
- Fréquence: Quotidien à 2h du matin
- Rétention: 7 jours en local
- Upload vers S3 (optionnel): Configurer `BACKUP_S3_BUCKET` dans .env

### Backups manuels
```bash
# Sur l'instance EC2
ssh ec2-user@<instance_ip>
cd /opt/harena

# Backup PostgreSQL
docker-compose exec -T postgres pg_dump -U harena_admin harena > backup_$(date +%Y%m%d).sql

# Backup Elasticsearch
curl -X PUT "http://localhost:9200/_snapshot/backup_repo/snapshot_$(date +%Y%m%d)?wait_for_completion=true"
```

### Restauration depuis backup
```bash
# PostgreSQL
docker-compose exec -T postgres psql -U harena_admin harena < backup_20250112.sql

# Elasticsearch
curl -X POST "http://localhost:9200/_snapshot/backup_repo/snapshot_20250112/_restore"
```

## Monitoring des coûts

### Coûts par service (estimés)
```
EC2 t4g.small Spot (730h × $0.0084/h)  : $6.13/mois
EBS gp3 40 GB ($0.08/GB)               : $3.20/mois
S3 storage (~1GB)                      : $0.02/mois
S3 requests                            : $0.05/mois
CloudFront (faible trafic)             : $1.00/mois
CloudWatch Logs (7 jours retention)    : $0.50/mois
Data transfer (5GB/mois)               : $0.43/mois
---------------------------------------------------
TOTAL                                  : ~$11.33/mois
```

### Optimisations supplémentaires possibles

1. **Auto-shutdown** (déjà configuré):
   - Arrêt nocturne (22h-8h): Économie de ~50h/semaine = $2.52/mois
   - Arrêt weekend (samedi-dimanche): Économie de ~96h/mois = $4.84/mois
   - **Total économies**: ~$7.36/mois → Coût final: **~$4/mois**

2. **Reserved Instance** (si utilisation 24/7):
   - t4g.small RI 1 an: $3.85/mois (au lieu de $6.13)
   - Économie: $2.28/mois

3. **Réduire les logs CloudWatch**:
   - Passer de 7 jours à 3 jours de rétention
   - Économie: ~$0.20/mois

## Troubleshooting

### PostgreSQL ne démarre pas
```bash
# Vérifier les logs
docker-compose logs postgres

# Vérifier les permissions
docker-compose exec postgres ls -la /var/lib/postgresql/data

# Recréer le volume
docker-compose down -v
docker-compose up -d postgres
```

### Elasticsearch ne démarre pas
```bash
# Vérifier les logs
docker-compose logs elasticsearch

# Vérifier la mémoire disponible
free -h

# Réduire la mémoire allouée si nécessaire
# Éditer docker-compose.yml: ES_JAVA_OPTS=-Xms256m -Xmx256m
```

### Backend ne se connecte pas aux services
```bash
# Vérifier que tous les services sont démarrés
docker-compose ps

# Vérifier la connectivité réseau
docker-compose exec backend ping postgres
docker-compose exec backend ping redis
docker-compose exec backend ping elasticsearch

# Vérifier les variables d'environnement
docker-compose exec backend env | grep -E "(DB_|REDIS_|ELASTICSEARCH_)"
```

### Spot instance interrompue
Les instances Spot peuvent être interrompues si le prix dépasse le max_price configuré.

**Solution**:
- AWS enverra un avertissement 2 minutes avant l'interruption
- Les données sur les volumes EBS sont conservées
- L'instance redémarrera automatiquement quand le prix baisse
- Pour éviter: passer en instance On-Demand (coût: ~$12/mois au lieu de $6)

## Support

En cas de problème:
1. Consultez les logs: `docker-compose logs -f`
2. Vérifiez le health check: `curl http://localhost:8001/health`
3. Utilisez le script de rollback si nécessaire
4. Conservez TOUS les backups pendant au moins 30 jours

## Checklist post-migration

- [ ] Tous les services Docker sont running
- [ ] Les données PostgreSQL sont migrées
- [ ] Les indices Elasticsearch sont présents
- [ ] L'API répond correctement
- [ ] Le frontend est mis à jour avec la nouvelle URL
- [ ] Les backups automatiques fonctionnent
- [ ] Le monitoring CloudWatch est configuré
- [ ] Les alertes sont configurées (optionnel)
- [ ] L'auto-shutdown est testé (optionnel)
- [ ] Les coûts AWS sont surveillés
- [ ] Ancienne infrastructure détruite (après validation)
