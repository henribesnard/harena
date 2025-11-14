# Infrastructure AWS - Harena

Documentation complète de l'infrastructure AWS et procédures de déploiement pour le projet Harena.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Accès à l'infrastructure](#accès-à-linfrastructure)
3. [Configuration des services](#configuration-des-services)
4. [Procédures de déploiement](#procédures-de-déploiement)
5. [Monitoring et logs](#monitoring-et-logs)
6. [Résolution de problèmes](#résolution-de-problèmes)
7. [Sécurité](#sécurité)

---

## Vue d'ensemble

### Architecture déployée

L'application Harena est déployée sur AWS avec l'architecture suivante :

```
┌─────────────────────────────────────────────────────────────┐
│                         Internet                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  EC2 Instance │
                  │ 63.35.52.216 │
                  └──────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Nginx   │    │  Docker │    │  RDS    │
    │ :80     │    │ Network │    │ :5432   │
    └─────────┘    └────┬────┘    └─────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐    ┌───▼────┐    ┌───▼────┐
    │  User   │    │ Search │    │ Sync   │
    │ Service │    │Service │    │Service │
    │  :3000  │    │ :3001  │    │ :3004  │
    └─────────┘    └────────┘    └────────┘
         │              │              │
    ┌────▼────┐    ┌───▼────┐    ┌───▼────────┐
    │ Metric  │    │Budget  │    │Enrichment  │
    │ Service │    │Profiling│   │  Service   │
    │  :3002  │    │ :3006  │    │   :3005    │
    └─────────┘    └────────┘    └────────────┘
         │
    ┌────▼──────────┐
    │ Conversation  │
    │  Service V3   │
    │    :3008      │
    └───────────────┘
```

### Informations clés

| Ressource | Valeur | Description |
|-----------|--------|-------------|
| **Instance EC2** | `i-0011b978b7cea66dc` | Instance principale hébergeant les services Docker |
| **IP Publique** | `63.35.52.216` | IP publique fixe (Elastic IP) |
| **URL Cloudflare** | `https://food-dining-email-riders.trycloudflare.com` | URL publique gratuite (temporaire, phase dev) |
| **Région AWS** | `eu-west-1` | Europe (Irlande) |
| **Zone** | `eu-west-1a` | Zone de disponibilité |
| **Security Group** | `sg-0aa65b430c3e93bad` | Groupe de sécurité `harena-allinone-sg-dev` |
| **IAM Role** | `harena-ec2-profile-dev` | Profil IAM avec accès SSM |
| **Base de données** | `63.35.52.216:5432` | PostgreSQL (RDS ou sur EC2) |
| **Redis** | `63.35.52.216:6379` | Cache Redis |

**Note** : L'URL Cloudflare est active uniquement quand le tunnel `cloudflared` est en cours d'exécution sur l'instance EC2.

---

## Accès à l'infrastructure

### Accès public via Cloudflare Tunnel

**URL publique** : `https://food-dining-email-riders.trycloudflare.com`

Cloudflare Tunnel expose Harena sur internet de manière sécurisée (SSL automatique, protection DDoS) sans ouvrir de ports supplémentaires.

**Vérifier si le tunnel est actif :**
```bash
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["ps aux | grep cloudflared | grep -v grep"]' \
    --region eu-west-1
```

**Voir l'URL actuelle du tunnel :**
```bash
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["cat /home/ec2-user/cloudflare-tunnel.log | grep trycloudflare.com | tail -1"]' \
    --region eu-west-1
```

**Redémarrer le tunnel (nouvelle URL générée) :**
```bash
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["pkill cloudflared","nohup cloudflared tunnel --url http://localhost:80 > /home/ec2-user/cloudflare-tunnel.log 2>&1 &","sleep 5","cat /home/ec2-user/cloudflare-tunnel.log | grep trycloudflare.com"]' \
    --region eu-west-1
```

**Arrêter le tunnel :**
```bash
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["pkill cloudflared"]' \
    --region eu-west-1
```

**Note** : L'URL générée change à chaque redémarrage du tunnel. Pour une URL fixe, il faudrait acheter un domaine et configurer un tunnel nommé Cloudflare.

### Prérequis

1. **AWS CLI** installé et configuré
```bash
aws --version
aws configure
```

2. **Clé SSH** : `~/.ssh/harena-deploy-key.pem`
```bash
chmod 400 ~/.ssh/harena-deploy-key.pem
```

3. **Droits AWS** : Accès au compte AWS ID `204093577928`

### Méthodes d'accès

#### 1. Via AWS Systems Manager (SSM) - RECOMMANDÉ

**Avantages** : Pas besoin de clé SSH, logs centralisés, plus sécurisé

```bash
# Envoyer une commande
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker ps"]' \
    --region eu-west-1 \
    --output json

# Récupérer le résultat (remplacer COMMAND_ID)
aws ssm get-command-invocation \
    --command-id <COMMAND_ID> \
    --instance-id i-0011b978b7cea66dc \
    --region eu-west-1 \
    --query 'StandardOutputContent' \
    --output text
```

**Script d'aide** : Voir `aws-ssm-helper.sh` (à créer si nécessaire)

#### 2. Via SSH (Fallback)

```bash
ssh -i ~/.ssh/harena-deploy-key.pem ec2-user@63.35.52.216
```

**Note** : L'agent SSM doit être installé et configuré sur l'instance.

### Commandes utiles

```bash
# Vérifier l'état de l'instance
aws ec2 describe-instances \
    --instance-ids i-0011b978b7cea66dc \
    --region eu-west-1 \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]'

# Vérifier l'agent SSM
aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=i-0011b978b7cea66dc" \
    --region eu-west-1

# Lister les security groups
aws ec2 describe-security-groups \
    --group-ids sg-0aa65b430c3e93bad \
    --region eu-west-1 \
    --query 'SecurityGroups[0].IpPermissions[*].[IpProtocol,FromPort,ToPort]'
```

---

## Configuration des services

### Services déployés

| Service | Port | Conteneur Docker | Status |
|---------|------|------------------|--------|
| **Nginx** | 80 | `harena_nginx` | ✓ |
| **User Service** | 3000 | `harena_user_service` | ✓ |
| **Search Service** | 3001 | `harena_search_service` | ✓ |
| **Metric Service** | 3002 | `harena_metric_service` | ✓ |
| **Sync Service** | 3004 | `harena_sync_service` | ✓ |
| **Enrichment Service** | 3005 | `harena_enrichment_service` | ✓ |
| **Budget Profiling** | 3006 | `harena_budget_profiling_service` | ✓ |
| **Conversation V3** | 3008 | `harena_conversation_v3` | ✓ |
| **Grafana** | 4000 | `harena_grafana` | ✓ |
| **Prometheus** | 9090 | `harena_prometheus` | ✓ |

### Variables d'environnement (.env sur AWS)

**Localisation** : `/home/ec2-user/harena/.env`

**Variables critiques** :

```bash
# Base de données PostgreSQL
DATABASE_URL=postgresql://harena_admin:HaReNa2024SecureDbPassword123@63.35.52.216:5432/harena

# Redis
REDIS_URL=redis://:HaReNa2024-Redis-Auth-Token-Secure-Key-123456@63.35.52.216:6379/0

# Bridge API (sync bancaire)
BRIDGE_CLIENT_ID=<secret>
BRIDGE_CLIENT_SECRET=<secret>
BRIDGE_API_URL=https://api.bridgeapi.io
BRIDGE_API_VERSION=2021-06-01
BRIDGE_WEBHOOK_SECRET=<secret>

# Elasticsearch (enrichment)
ELASTICSEARCH_URL=http://elasticsearch:9200

# OpenAI / LLM
OPENAI_API_KEY=<secret>
LLM_MODEL=gpt-4o-mini
LLM_RESPONSE_MODEL=gpt-4o

# DeepSeek (alternative LLM)
DEEPSEEK_API_KEY=<secret>
DEEPSEEK_API_URL=https://api.deepseek.com
```

**⚠️ IMPORTANT** :
- Le `DATABASE_URL` doit utiliser le nom de conteneur Docker `harena_postgres` (pas l'IP `63.35.52.216`)
- Les services backend doivent communiquer via le réseau Docker interne, pas via l'IP publique
- Vérifier que le `.env` est à jour après chaque déploiement

### Ports ouverts dans le Security Group

```bash
# Vérifier les ports ouverts
aws ec2 describe-security-groups \
    --group-ids sg-0aa65b430c3e93bad \
    --region eu-west-1 \
    --query 'SecurityGroups[0].IpPermissions[*].[FromPort,ToPort,IpRanges[0].Description]' \
    --output table
```

**Ports actuellement ouverts** :
- 22 (SSH)
- 80 (HTTP - Nginx)
- 443 (HTTPS - non configuré pour l'instant)
- 3000 (User Service)
- 3001 (Search Service)
- 3002 (Metric Service)
- 3004 (Sync Service)
- 3005 (Enrichment Service)
- 3006 (Budget Profiling)
- 3008 (Conversation V3)
- 5432 (PostgreSQL)
- 6379 (Redis)

**Pour ajouter un port** :

```bash
# Exemple : Ouvrir le port 8080
aws ec2 authorize-security-group-ingress \
    --group-id sg-0aa65b430c3e93bad \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0 \
    --region eu-west-1 \
    --description "Description du service"
```

---

## Procédures de déploiement

### Déploiement manuel via SSM

#### 1. Mettre à jour le code sur AWS

```bash
# Via SSM
COMMAND_ID=$(aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["cd /home/ec2-user/harena && git pull origin main"]' \
    --region eu-west-1 \
    --output text --query 'Command.CommandId')

# Attendre et récupérer le résultat
sleep 5
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id i-0011b978b7cea66dc \
    --region eu-west-1 \
    --query 'StandardOutputContent' --output text
```

#### 2. Reconstruire et redémarrer les services

```bash
# Rebuild d'un service spécifique (ex: sync_service)
COMMAND_ID=$(aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["cd /home/ec2-user/harena && docker-compose build sync_service && docker-compose up -d --force-recreate --no-deps sync_service"]' \
    --region eu-west-1 \
    --output text --query 'Command.CommandId')

sleep 10
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id i-0011b978b7cea66dc \
    --region eu-west-1 \
    --query 'StandardOutputContent' --output text
```

#### 3. Vérifier le déploiement

```bash
# Vérifier que tous les conteneurs sont UP
COMMAND_ID=$(aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker ps --format \"table {{.Names}}\t{{.Status}}\t{{.Ports}}\""]' \
    --region eu-west-1 \
    --output text --query 'Command.CommandId')

sleep 3
aws ssm get-command-invocation \
    --command-id $COMMAND_ID \
    --instance-id i-0011b978b7cea66dc \
    --region eu-west-1 \
    --query 'StandardOutputContent' --output text
```

### Déploiement du frontend

**Script existant** : `update-frontend-aws.sh`

```bash
#!/bin/bash
# Déployer le frontend sur AWS

# 1. Build local
cd harena_front
npm run build

# 2. Transférer vers AWS via SCP
scp -r -i ~/.ssh/harena-deploy-key.pem dist/* ec2-user@63.35.52.216:/home/ec2-user/harena/harena_front/dist/

# 3. Redémarrer nginx
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker-compose restart nginx"]' \
    --region eu-west-1
```

### Rollback rapide

En cas de problème après déploiement :

```bash
# 1. Revenir à la version précédente du code
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["cd /home/ec2-user/harena && git reset --hard HEAD~1 && docker-compose up -d --force-recreate"]' \
    --region eu-west-1

# 2. Vérifier les logs
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker logs <service_name> --tail 50"]' \
    --region eu-west-1
```

---

## Monitoring et logs

### Grafana

**URL** : `http://63.35.52.216:4000`
**Credentials** : Voir `.env` (GRAFANA_ADMIN_USER / GRAFANA_ADMIN_PASSWORD)

**Dashboards disponibles** :
- Harena System Overview (`monitoring/grafana/dashboards/json/harena-overview.json`)

### Prometheus

**URL** : `http://63.35.52.216:9090`

**Métriques collectées** :
- Container metrics (CPU, RAM, Network)
- Node metrics (system-level)
- Application metrics (via exporters)

### Consulter les logs

#### Via SSM (depuis votre machine)

```bash
# Logs d'un service spécifique
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["docker logs harena_sync_service --tail 100"]' \
    --region eu-west-1

# Logs en temps réel (via SSH recommandé)
ssh -i ~/.ssh/harena-deploy-key.pem ec2-user@63.35.52.216
docker logs -f harena_sync_service
```

#### Via SSH

```bash
# Se connecter
ssh -i ~/.ssh/harena-deploy-key.pem ec2-user@63.35.52.216

# Logs de tous les services
cd /home/ec2-user/harena
docker-compose logs --tail=50

# Logs d'un service spécifique
docker logs harena_user_service --tail 100 -f

# Filtrer les erreurs
docker logs harena_sync_service 2>&1 | grep -i error
```

### Health checks

Tous les services exposent un endpoint `/health` :

```bash
# User Service
curl http://63.35.52.216:3000/health

# Sync Service
curl http://63.35.52.216:3004/health

# Conversation V3
curl http://63.35.52.216:3008/health
```

---

## Résolution de problèmes

### Service ne démarre pas

**1. Vérifier les logs**
```bash
docker logs harena_<service_name> --tail 50
```

**2. Vérifier les variables d'environnement**
```bash
docker exec harena_<service_name> env | grep DATABASE_URL
```

**3. Vérifier la connexion à la base de données**
```bash
# Depuis l'instance EC2
psql postgresql://harena_admin:PASSWORD@63.35.52.216:5432/harena -c "SELECT 1;"
```

### Problème de base de données

**Symptôme** : Erreur `could not translate host name "harena-postgres"`

**Cause** : Le `.env` sur AWS utilise le nom Docker local au lieu de l'IP

**Solution** :
```bash
# Corriger le DATABASE_URL
aws ssm send-command \
    --instance-ids i-0011b978b7cea66dc \
    --document-name "AWS-RunShellScript" \
    --parameters 'commands=["cd /home/ec2-user/harena && sed -i \"s/@harena-postgres:/@63.35.52.216:/g\" .env && docker-compose up -d --force-recreate"]' \
    --region eu-west-1
```

### Service timeout ou lent

**1. Vérifier les ressources**
```bash
docker stats
```

**2. Vérifier la configuration des pools de connexions**
- Voir `db_service/session.py` : `pool_size=20`, `max_overflow=30`

**3. Redémarrer le service**
```bash
docker-compose restart <service_name>
```

### Webhook Bridge ne fonctionne pas

**URL attendue** : `http://63.35.52.216:3004/webhooks/bridge`

**⚠️ Bridge exige HTTPS** - Actuellement non configuré

**Options** :
1. Configurer Let's Encrypt + nginx (nécessite un nom de domaine)
2. Utiliser CloudFlare Tunnel (gratuit)
3. Utiliser ngrok en local pour le développement

**Tester le webhook** :
```bash
curl -X POST http://63.35.52.216:3004/webhooks/bridge \
  -H "Content-Type: application/json" \
  -H "BridgeApi-Signature: test" \
  -d '{"type": "test", "content": {}}'
```

---

## Sécurité

### Credentials et secrets

**⚠️ NE JAMAIS COMMITER** :
- `.env` (local et production)
- Clés SSH
- Tokens d'API

**Bonnes pratiques** :
1. Utiliser AWS Secrets Manager pour les secrets sensibles
2. Rotation régulière des mots de passe
3. Principe du moindre privilège pour les IAM roles

### Accès SSH

**Désactivé pour l'instant** - Utiliser SSM en priorité

**Si SSH nécessaire** :
```bash
ssh -i ~/.ssh/harena-deploy-key.pem ec2-user@63.35.52.216
```

**Permissions de la clé** :
```bash
chmod 400 ~/.ssh/harena-deploy-key.pem
```

### Firewall (Security Group)

**Règles actuelles** : Tous les ports ouverts à `0.0.0.0/0` (internet)

**⚠️ Recommandation** : Restreindre l'accès aux ports sensibles :
- PostgreSQL (5432) : Accès limité à l'IP de l'instance uniquement
- Redis (6379) : Accès limité à l'IP de l'instance uniquement
- Services backend (3000-3008) : Accès via nginx uniquement (fermer les ports publics)

**Exemple de restriction** :
```bash
# Supprimer la règle actuelle
aws ec2 revoke-security-group-ingress \
    --group-id sg-0aa65b430c3e93bad \
    --protocol tcp --port 5432 --cidr 0.0.0.0/0 \
    --region eu-west-1

# Ajouter une règle restreinte
aws ec2 authorize-security-group-ingress \
    --group-id sg-0aa65b430c3e93bad \
    --protocol tcp --port 5432 \
    --source-group sg-0aa65b430c3e93bad \
    --region eu-west-1
```

### Backup

**⚠️ À CONFIGURER** :

1. **Snapshots EC2** : Sauvegardes régulières de l'instance
```bash
aws ec2 create-snapshot \
    --volume-id <volume-id> \
    --description "Harena backup $(date +%Y-%m-%d)" \
    --region eu-west-1
```

2. **Backup PostgreSQL** :
```bash
# Via pg_dump sur l'instance
ssh ec2-user@63.35.52.216
pg_dump postgresql://harena_admin:PASSWORD@localhost:5432/harena > backup_$(date +%Y%m%d).sql

# Télécharger le backup
scp -i ~/.ssh/harena-deploy-key.pem \
    ec2-user@63.35.52.216:/home/ec2-user/backup_*.sql \
    ./backups/
```

3. **RDS Automated Backups** : Si PostgreSQL est sur RDS (à vérifier)

---

## Checklist de déploiement

### Avant de déployer

- [ ] Tests locaux réussis
- [ ] Code commité et poussé sur `main`
- [ ] `.env` à jour sur AWS
- [ ] Backup de la base de données effectué
- [ ] Grafana/Prometheus fonctionnels

### Déploiement

- [ ] Pull du code sur AWS
- [ ] Rebuild des images Docker modifiées
- [ ] Redémarrage des services concernés
- [ ] Vérification des logs (pas d'erreurs)
- [ ] Tests des endpoints critiques
- [ ] Vérification du dashboard Grafana

### Après déploiement

- [ ] Monitoring actif (Grafana)
- [ ] Webhooks Bridge testés (si applicable)
- [ ] Notifier l'équipe du déploiement
- [ ] Documenter les changements

---

## Contacts et ressources

### Documentation technique

- **Docker Compose** : `docker-compose.yml`
- **Nginx** : `nginx/conf.d/harena.conf`
- **Monitoring** : `monitoring/`
- **Scripts** : Voir `*.sh` à la racine

### Ressources AWS

- **Console AWS** : https://console.aws.amazon.com
- **Région** : eu-west-1 (Irlande)
- **Compte ID** : 204093577928

### Support

- **Documentation Bridge API** : https://docs.bridgeapi.io
- **Documentation AWS SSM** : https://docs.aws.amazon.com/systems-manager/

---

## Changelog Infrastructure

### 2025-11-06
- ✅ Installation de Cloudflare Tunnel (cloudflared) sur EC2 (ARM64)
  - **URL publique** : `https://food-dining-email-riders.trycloudflare.com`
  - SSL automatique, protection DDoS gratuite
  - Tunnel gratuit temporaire pour phase de développement
  - Commandes de gestion documentées
- ✅ Documentation complète de l'accès via Cloudflare Tunnel

### 2025-10-30
- ✅ Correction critique du `DATABASE_URL` dans `.env` sur AWS (`63.35.52.216` → `harena_postgres`)
  - **Problème** : Les services essayaient de se connecter via l'IP publique au lieu du réseau Docker
  - **Solution** : Utilisation du nom de conteneur Docker `harena_postgres` pour connexion interne
  - **Impact** : Tous les services backend maintenant "healthy" (user, search, metric, sync, enrichment, budget_profiling)
- ✅ Correction de la configuration Loki (arrêt des crashs en boucle)
  - Migration de `boltdb-shipper` (v11) vers `tsdb` (v13)
  - Suppression des champs obsolètes (`shared_store`, `max_look_back_period`, `table_manager`)
  - Compatible avec Loki 2.9+
- ✅ Ajout du endpoint `/health` au `search_service`
  - Corrige le statut "unhealthy" du healthcheck Docker
- ✅ Redéploiement complet de tous les services backend avec la nouvelle configuration

### 2025-10-29
- ✅ Première tentative de correction du `DATABASE_URL` (incorrecte)
- ✅ Redémarrage de `sync_service` et `enrichment_service`
- ✅ Validation des webhooks Bridge (endpoint fonctionnel mais HTTPS requis)
- ✅ Documentation complète de l'infrastructure créée

### À faire
- [ ] Configurer HTTPS avec Let's Encrypt ou CloudFlare Tunnel
- [ ] Restreindre les security groups (principe du moindre privilège)
- [ ] Mettre en place des backups automatisés
- [ ] Configurer des alertes Grafana
- [ ] Elastic IP pour l'instance EC2
- [ ] Domain name + certificat SSL

---

**Document maintenu par** : Équipe Harena
**Dernière mise à jour** : 2025-11-06
