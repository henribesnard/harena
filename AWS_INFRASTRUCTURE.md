# Infrastructure Backend AWS - Harena

## Vue d'ensemble

L'infrastructure backend de Harena est déployée sur AWS EC2 avec une architecture microservices utilisant Docker et Nginx comme reverse proxy.

**Date de mise à jour** : 24 Octobre 2025

---

## Instance AWS EC2

### Informations principales

- **Instance ID** : `i-0011b978b7cea66dc`
- **Nom** : `harena-allinone-dev`
- **IP Publique** : `63.35.52.216` (Elastic IP)
- **IP Privée** : `10.0.1.80`
- **Région** : `eu-west-1` (Ireland)
- **Type d'instance** : t3.medium (ou similaire)
- **OS** : Amazon Linux 2023

### Security Group

- **Group ID** : `sg-0aa65b430c3e93bad`
- **Nom** : `harena-allinone-sg-dev`

**Ports ouverts** :
- `80` (HTTP) - Nginx reverse proxy - `0.0.0.0/0`
- `443` (HTTPS) - SSL/TLS - `0.0.0.0/0`
- `3000` (User Service) - Accès direct - `0.0.0.0/0`
- `3001` (Search Service) - Accès direct - `0.0.0.0/0`
- `3002` (Metric Service) - Accès direct - `0.0.0.0/0`
- `3006` (Budget Profiling Service) - Accès direct - `0.0.0.0/0`
- `3008` (Conversation Service V3) - Accès direct - `0.0.0.0/0`
- `5432` (PostgreSQL) - Accès restreint IP spécifique
- `6379` (Redis) - Accès restreint IP spécifique

---

## Services Backend Déployés

### 1. User Service (Port 3000)

**Description** : Gestion des utilisateurs, authentification et connexions bancaires Bridge API

**URLs** :
- **Accès direct** : `http://63.35.52.216:3000`
- **Via Nginx** : `http://63.35.52.216/api/v1/users`, `/api/v1/auth`
- **Health check** : `http://63.35.52.216:3000/health`
- **Documentation** : `http://63.35.52.216:3000/docs`

**Endpoints principaux** :
- `POST /api/v1/users/auth/login` - Authentification (OAuth2PasswordRequestForm)
- `GET /api/v1/users/me` - Profil utilisateur
- `GET /api/v1/users/bridge/accounts` - Comptes bancaires

**Container** : `harena_user_service_prod`

---

### 2. Search Service (Port 3001)

**Description** : Recherche et indexation de transactions via Elasticsearch/Bonsai

**URLs** :
- **Accès direct** : `http://63.35.52.216:3001`
- **Via Nginx** : `http://63.35.52.216/api/v1/search`
- **Health check** : `http://63.35.52.216:3001/api/v1/search/health`
- **Documentation** : `http://63.35.52.216:3001/docs`

**Endpoints principaux** :
- `POST /api/v1/search/transactions` - Recherche de transactions
- `POST /api/v1/search/index` - Indexation de transactions
- `GET /api/v1/search/health` - Statut du service

**Container** : `harena_search_service_prod`

---

### 3. Metric Service (Port 3002)

**Description** : Métriques financières et analytics avec prévisions Prophet

**URLs** :
- **Accès direct** : `http://63.35.52.216:3002`
- **Via Nginx** : `http://63.35.52.216/api/v1/metrics`
- **Health check** : `http://63.35.52.216:3002/health`
- **Documentation** : `http://63.35.52.216:3002/docs`

**Endpoints principaux** :
- `GET /api/v1/metrics/expenses/mom` - Dépenses mois par mois
- `GET /api/v1/metrics/expenses/yoy` - Dépenses année par année
- `GET /api/v1/metrics/income/mom` - Revenus mois par mois
- `GET /api/v1/metrics/income/yoy` - Revenus année par année
- `GET /api/v1/metrics/coverage` - Taux de couverture

**Container** : `harena_metric_service_prod`

---

### 4. Budget Profiling Service (Port 3006)

**Description** : Analyse budgétaire et recommandations intelligentes

**URLs** :
- **Accès direct** : `http://63.35.52.216:3006`
- **Via Nginx** : `http://63.35.52.216/api/v1/budget`
- **Health check** : `http://63.35.52.216:3006/health`
- **Documentation** : `http://63.35.52.216:3006/docs`

**Endpoints principaux** :
- `GET /api/v1/budget/profiles` - Profils budgétaires
- `POST /api/v1/budget/analyze` - Analyse de budget
- `GET /api/v1/budget/recommendations` - Recommandations

**Container** : `harena_budget_profiling_prod`

---

### 5. Conversation Service V3 (Port 3008)

**Description** : Service conversationnel IA avec agents LangChain et support WebSocket

**URLs** :
- **Accès direct** : `http://63.35.52.216:3008`
- **Via Nginx** : `http://63.35.52.216/api/v3`
- **Health check** : `http://63.35.52.216:3008/health`
- **Documentation** : `http://63.35.52.216:3008/docs`

**Endpoints principaux** :
- `POST /api/v3/query` - Requête conversationnelle
- `GET /api/v3/conversations/{user_id}` - Historique de conversations
- `WebSocket /api/v3/ws` - Streaming en temps réel

**Container** : `harena_conversation_v3_prod`

**Fonctionnalités** :
- Agent LangChain pour requêtes Elasticsearch
- Correction automatique de requêtes (max 2 tentatives)
- Support WebSocket pour streaming de réponses
- Modèles LLM : GPT-4o-mini (requêtes), GPT-4o (réponses)

---

## Infrastructure de Données

### PostgreSQL

- **Container** : `harena-postgres`
- **Port** : `5432`
- **Accès** : Via réseau Docker `harena-network-prod`
- **URL interne** : `postgresql://harena_admin:***@harena-postgres:5432/harena`
- **Statut** : Healthy (Up 7+ days)

### Redis

- **Container** : `harena-redis`
- **Port** : `6379`
- **Accès** : Via réseau Docker `harena-network-prod`
- **URL interne** : `redis://:***@harena-redis:6379/0`
- **Statut** : Healthy (Up 7+ days)

### Elasticsearch

- **Provider** : Heroku Bonsai
- **URL** : `https://37r8v9zfzn:***@fir-178893546.eu-west-1.bonsaisearch.net:443`
- **Accès** : Externe (cloud)

---

## Nginx Reverse Proxy

### Configuration

- **Port** : `80` (HTTP)
- **Fichier de config** : `/etc/nginx/conf.d/harena.conf`

### Routing

Nginx route automatiquement les requêtes vers les services appropriés :

| Route | Service | Port Backend |
|-------|---------|--------------|
| `/api/v1/users`, `/api/v1/auth` | User Service | 3000 |
| `/api/v1/search` | Search Service | 3001 |
| `/api/v1/metrics` | Metric Service | 3002 |
| `/api/v1/budget` | Budget Profiling Service | 3006 |
| `/api/v3` | Conversation Service V3 | 3008 |

### Fonctionnalités Nginx

- **CORS** : Géré centralement (désactivé dans les services)
- **Rate limiting** : 30 req/s général, 10 req/s conversation
- **WebSocket** : Support activé pour `/api/v3`
- **Timeouts** : 60s standard, 86400s (24h) pour WebSocket
- **Security headers** : X-Content-Type-Options, X-Frame-Options, X-XSS-Protection

### Documentation centralisée

- **Index des services** : `http://63.35.52.216/docs`

---

## Réseau Docker

### harena-network-prod

- **Type** : Bridge
- **Subnet** : `172.20.0.0/16`
- **Gateway** : `172.20.0.1`

**Conteneurs connectés** :
- harena_user_service_prod
- harena_search_service_prod
- harena_metric_service_prod
- harena_budget_profiling_prod
- harena_conversation_v3_prod
- harena-postgres (multi-réseau)
- harena-redis (multi-réseau)

---

## Configuration Frontend

### Variables d'environnement (.env.production / .env.aws)

```bash
VITE_USER_SERVICE_URL=http://63.35.52.216
VITE_SEARCH_SERVICE_URL=http://63.35.52.216
VITE_METRIC_SERVICE_URL=http://63.35.52.216
VITE_CONVERSATION_V1_SERVICE_URL=http://63.35.52.216
VITE_CONVERSATION_SERVICE_URL=http://63.35.52.216
VITE_SYNC_SERVICE_URL=http://63.35.52.216
VITE_ENRICHMENT_SERVICE_URL=http://63.35.52.216
VITE_BUDGET_PROFILING_API_URL=http://63.35.52.216
```

**Note** : Tous les services utilisent la même IP publique. Nginx route automatiquement selon le path.

---

## Déploiement et Maintenance

### Scripts de déploiement

- `deploy.sh` - Script principal de déploiement
- `deploy-nginx.sh` - Déploiement de la configuration Nginx
- `pre-deploy-check.sh` - Vérifications pré-déploiement

### Commandes Docker utiles

```bash
# Voir les conteneurs
docker ps

# Voir les logs d'un service
docker logs -f harena_user_service_prod

# Redémarrer un service
docker-compose -f docker-compose.prod.yml restart user_service

# Redémarrer tous les services
docker-compose -f docker-compose.prod.yml restart

# Rebuild et redémarrer
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Voir les ressources
docker stats
```

### Commandes AWS SSM

```bash
# Exécuter une commande sur l'instance
aws ssm send-command \
  --instance-ids i-0011b978b7cea66dc \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["<commande>"]'

# Voir le résultat
aws ssm get-command-invocation \
  --command-id <command-id> \
  --instance-id i-0011b978b7cea66dc
```

---

## Tests et Monitoring

### Health checks

Tous les services exposent un endpoint `/health` :

```bash
curl http://63.35.52.216:3000/health
curl http://63.35.52.216:3001/api/v1/search/health
curl http://63.35.52.216:3002/health
curl http://63.35.52.216:3006/health
curl http://63.35.52.216:3008/health
```

### Test d'authentification (Bruno/Postman)

**Endpoint** : `POST http://63.35.52.216/api/v1/users/auth/login`

**Headers** :
```
Content-Type: application/x-www-form-urlencoded
```

**Body (Form URL Encoded)** :
```
username=henri@example.com
password=Henri123456
```

**Réponse attendue** :
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

### Monitoring des ressources

```bash
# Sur l'instance EC2
docker stats
free -h
df -h
```

---

## Architecture et Flux de données

```
┌─────────────────────────────────────────────────────────┐
│                    Internet (Client)                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ HTTP/HTTPS
                         ▼
┌─────────────────────────────────────────────────────────┐
│              AWS EC2 - 63.35.52.216:80                  │
│                   Nginx Reverse Proxy                    │
│         (CORS, Rate Limiting, Load Balancing)           │
└───┬──────────┬──────────┬──────────┬──────────┬─────────┘
    │          │          │          │          │
    ├──────────┼──────────┼──────────┼──────────┤
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌──────────┐
│ User  │ │ Search │ │ Metric  │ │ Budget │ │ Conv. V3 │
│ :3000 │ │ :3001  │ │ :3002   │ │ :3006  │ │ :3008    │
└───┬───┘ └───┬────┘ └────┬────┘ └───┬────┘ └────┬─────┘
    │         │           │          │           │
    └─────────┴───────────┴──────────┴───────────┘
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌─────────────────┐
    │   PostgreSQL     │    │   Redis Cache   │
    │   harena-postgres│    │   harena-redis  │
    │      :5432       │    │      :6379      │
    └──────────────────┘    └─────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  Elasticsearch   │
    │  Heroku Bonsai   │
    │  (External)      │
    └──────────────────┘
```

---

## Troubleshooting

### Les services ne répondent pas

1. Vérifier que les conteneurs sont up :
   ```bash
   docker ps
   ```

2. Vérifier les logs :
   ```bash
   docker logs harena_user_service_prod
   ```

3. Vérifier nginx :
   ```bash
   sudo systemctl status nginx
   sudo nginx -t
   ```

### Erreurs de connexion base de données

1. Vérifier que PostgreSQL est connecté au bon réseau :
   ```bash
   docker network inspect harena-network-prod
   ```

2. Vérifier la variable DATABASE_URL dans les conteneurs :
   ```bash
   docker exec harena_user_service_prod env | grep DATABASE_URL
   ```

### Erreurs CORS

- CORS est géré uniquement par Nginx
- Vérifier que CORSMiddleware est désactivé dans les services Python
- Les headers CORS ne doivent apparaître qu'une seule fois dans les réponses

---

## Sécurité

### Best Practices

- ✅ PostgreSQL et Redis accessibles uniquement via réseau Docker interne
- ✅ Security Group restreint pour PostgreSQL et Redis (IP spécifique)
- ✅ Services backend accessibles publiquement mais via Nginx (rate limiting)
- ✅ Headers de sécurité configurés dans Nginx
- ✅ Secrets stockés dans variables d'environnement (non versionnées)

### Points à améliorer

- [ ] Activer HTTPS/SSL avec Let's Encrypt
- [ ] Restreindre l'accès direct aux services (ports 3000-3008) en production
- [ ] Configurer un domaine personnalisé
- [ ] Mettre en place un système de monitoring (CloudWatch, Prometheus)
- [ ] Configurer des backups automatiques pour PostgreSQL

---

## Contacts et Support

**Repository** : [henribesnard/harena](https://github.com/henribesnard/harena)

**Documentation** :
- API Index : http://63.35.52.216/docs
- User Service : http://63.35.52.216:3000/docs
- Metric Service : http://63.35.52.216:3002/docs
- Budget Service : http://63.35.52.216:3006/docs
- Search Service : http://63.35.52.216:3001/docs
- Conversation V3 : http://63.35.52.216:3008/docs
