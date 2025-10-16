# Plan de Déploiement AWS - Harena v4.0.1

## 📦 État Actuel (Pré-Déploiement)

✅ **Backups complets créés** (85 MB, 25 fichiers)
✅ **Version v4.0.1 taggée et poussée**
✅ **Configuration locale documentée**

## 🎯 Objectifs

1. Déployer tous les services backend sur AWS
2. Utiliser le **port 8000 uniformément** pour tous les services
3. Configurer le frontend pour pointer vers l'URL AWS
4. Maintenir Elasticsearch (Bonsai) et PostgreSQL externes

## 🏗️ Architecture Cible AWS

### Services Backend (tous sur port 8000)
```
┌─────────────────────────────────────────┐
│          AWS EC2 / ECS / Lambda         │
├─────────────────────────────────────────┤
│                                         │
│  user_service          :8000            │
│  search_service        :8000            │
│  conversation_service  :8000            │
│                                         │
└─────────────────────────────────────────┘
         ↓                    ↓
   ┌──────────┐         ┌───────────┐
   │PostgreSQL│         │Elasticsearch│
   │External  │         │Bonsai       │
   │63.35.x.x │         │fir-178...   │
   └──────────┘         └───────────────┘
```

### Frontend
```
┌──────────────────┐
│  AWS S3 + CloudFront
│  ou Vercel/Netlify │
│                    │
│  harena_front      │
│                    │
└──────────────────┘
```

## 📝 Configuration des Ports

### ⚠️ IMPORTANT: Port 8000 Uniforme

Tous les services backend utiliseront le **même port 8000** pour simplicité:

**Option 1: Services séparés avec routing**
- Load Balancer AWS avec path-based routing:
  - `/api/user/*` → user_service:8000
  - `/api/search/*` → search_service:8000
  - `/api/conversation/*` → conversation_service:8000

**Option 2: Une seule instance avec tous les services**
- Les 3 services sur différents ports internes
- Exposer via nginx/reverse proxy sur port 8000

**Option 3 (Recommandée): Docker Compose sur EC2**
- Chaque service dans son conteneur Docker
- nginx reverse proxy unifiant sur port 8000
- docker-compose.yml gère l'orchestration

## 🔧 Modifications Nécessaires

### 1. Conversation Service (conversation_service/main.py)
**Changement:** Port 8001 → 8000

```python
# AVANT
uvicorn.run(..., port=8001, ...)

# APRÈS
uvicorn.run(..., port=8000, ...)
```

### 2. Frontend (harena_front/.env)
**Changement:** localhost → URL AWS

```env
# AVANT
VITE_API_URL=http://localhost:8000

# APRÈS
VITE_API_URL=https://api.harena.aws-domain.com
# OU
VITE_API_URL=http://ec2-XX-XXX-XXX-XXX.compute.amazonaws.com:8000
```

### 3. Fichiers Frontend à Mettre à Jour
- `harena_front/.env`
- `harena_front/src/services/api.ts`
- `harena_front/src/services/api/coreMetricsApi.ts`
- `harena_front/src/services/api/metricsApi.ts`

## 🔐 Variables d'Environnement AWS

### Secrets Manager / Parameter Store

```bash
# Base de données
DATABASE_URL=postgresql://harena_admin:PASSWORD@63.35.52.216:5432/harena

# Elasticsearch
BONSAI_URL=https://USER:PASS@fir-178893546.eu-west-1.bonsaisearch.net:443

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256

# LLM
DEEPSEEK_API_KEY=sk-xxx

# Services
SEARCH_SERVICE_URL=http://search-service:8000
USER_SERVICE_URL=http://user-service:8000
CONVERSATION_SERVICE_URL=http://conversation-service:8000
```

## 📦 Options de Déploiement

### Option A: EC2 avec Docker Compose (Recommandée)
**Avantages:**
- Simple et rapide
- Contrôle total
- Coût prévisible
- Facile à débugger

**Stack:**
- 1 EC2 t3.medium
- Docker + docker-compose
- nginx reverse proxy
- Let's Encrypt SSL

**Fichier:** `docker-compose.yml`

### Option B: ECS Fargate
**Avantages:**
- Serverless
- Scaling automatique
- Pas de gestion serveur

**Stack:**
- ECS Tasks pour chaque service
- Application Load Balancer
- CloudWatch Logs

### Option C: AWS Lambda + API Gateway
**Avantages:**
- Pay-per-use
- Scaling infini
- Pas de serveur

**Contraintes:**
- Froid démarrage
- Timeout 15min max
- Adapter le code

## 🚀 Plan de Déploiement Étape par Étape

### Phase 1: Préparation (Complété ✅)
- [x] Backups PostgreSQL
- [x] Backups Elasticsearch
- [x] Documentation configuration
- [x] Tag version v4.0.1

### Phase 2: Configuration Docker (À faire)
- [ ] Créer Dockerfile pour user_service
- [ ] Créer Dockerfile pour search_service
- [ ] Créer Dockerfile pour conversation_service
- [ ] Créer docker-compose.yml
- [ ] Créer nginx.conf pour reverse proxy
- [ ] Tester localement avec Docker

### Phase 3: Infrastructure AWS (À faire)
- [ ] Créer EC2 instance (t3.medium, Ubuntu 22.04)
- [ ] Configurer Security Groups (port 80, 443, 8000)
- [ ] Installer Docker + docker-compose
- [ ] Configurer domaine/DNS (Route53)
- [ ] Configurer SSL (Let's Encrypt)

### Phase 4: Déploiement Services (À faire)
- [ ] Cloner repo sur EC2
- [ ] Configurer variables d'environnement
- [ ] Build images Docker
- [ ] Lancer docker-compose up -d
- [ ] Vérifier santé des services

### Phase 5: Frontend (À faire)
- [ ] Mettre à jour VITE_API_URL
- [ ] Build production (npm run build)
- [ ] Déployer sur S3 + CloudFront ou Vercel
- [ ] Configurer domaine frontend

### Phase 6: Tests & Monitoring (À faire)
- [ ] Tests end-to-end
- [ ] Configurer CloudWatch logs
- [ ] Configurer alertes
- [ ] Tests de charge

## 📊 URLs Finales

**À définir après déploiement:**

```
Backend API: https://api.harena.com (ou http://XX.XXX.XXX.XXX:8000)
Frontend:    https://harena.com (ou https://harena.vercel.app)

Endpoints:
- https://api.harena.com/api/user/*
- https://api.harena.com/api/search/*
- https://api.harena.com/api/conversation/*
- https://api.harena.com/health
```

## 💰 Estimation des Coûts AWS

**Option EC2 (1 instance t3.medium):**
- Instance: ~$30/mois
- EBS 30GB: ~$3/mois
- Elastic IP: ~$3/mois si non attachée
- Total: ~$36/mois

**Option ECS Fargate:**
- 3 tasks 0.5 vCPU / 1GB: ~$20-40/mois
- Load Balancer: ~$16/mois
- Total: ~$36-56/mois

**Bases Externes (actuelles):**
- PostgreSQL: Déjà hébergé (63.35.52.216)
- Elasticsearch: Bonsai (déjà configuré)

## ⚠️ Points d'Attention

1. **Port Uniforme:** Tous services sur 8000 pour simplicité
2. **conversation_service:** Modifier port 8001 → 8000
3. **Frontend:** Mettre à jour 4 fichiers avec nouvelle URL
4. **Bases Externes:** PostgreSQL et Elasticsearch restent sur infra actuelle
5. **SSL/HTTPS:** Important pour production (Let's Encrypt gratuit)
6. **Monitoring:** CloudWatch ou équivalent obligatoire
7. **Backups:** Automatiser backups PostgreSQL (pg_dump cron)

## 📝 Prochaines Étapes

1. **Confirmer option de déploiement** (EC2, ECS, ou Lambda)
2. **Créer fichiers Docker**
3. **Provisionner infrastructure AWS**
4. **Déployer et tester**
5. **Communiquer URLs finales**

---

**Version:** v4.0.1
**Date:** 2025-10-16
**Status:** Prêt pour déploiement ✅
