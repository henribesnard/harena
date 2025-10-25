# 🚀 Quick Start - Harena Dev/Prod

## 📋 Prérequis

### Local (Dev)
- Docker Desktop installé et démarré
- Git
- Ports libres: 5174, 3000-3008, 5433, 6380, 9000

### AWS (Prod)
- Instance EC2: i-0011b978b7cea66dc
- Accès SSH ou AWS Systems Manager
- PostgreSQL et Redis déjà déployés

---

## 🟢 Démarrage DEV (Local)

### Première fois

```bash
# 1. Cloner le repo
git clone <votre-repo> harena
cd harena

# 2. Créer le fichier .env.dev
cp .env.dev.example .env.dev
# Éditer .env.dev avec vos clés API

# 3. Installer Portainer (une seule fois)
docker volume create portainer_data
docker run -d \
  -p 9000:9000 \
  --name portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest

# 4. Démarrer l'environnement
chmod +x start-dev.sh
./start-dev.sh
```

### Utilisation quotidienne

```bash
# Démarrer
./start-dev.sh

# Voir les logs
docker-compose -f docker-compose.dev.yml logs -f

# Arrêter
docker-compose -f docker-compose.dev.yml down

# Rebuild après changements
docker-compose -f docker-compose.dev.yml up -d --build
```

### Accès Dev

- **Frontend**: http://localhost:5174
- **Backend APIs**: http://localhost:3000-3008
- **PostgreSQL (DBeaver)**: localhost:5433
  - Database: `harena_dev`
  - User: `harena_dev`
  - Password: `dev_password_123`
- **Portainer**: http://localhost:9000

---

## 🔴 Déploiement PROD (AWS)

### Première fois

```bash
# 1. Connexion SSH à l'instance
aws ssm start-session --region eu-west-1 --target i-0011b978b7cea66dc

# 2. Cloner/Pull le code
cd /home/ec2-user
git clone <votre-repo> harena
cd harena

# 3. Créer le fichier de marqueur AWS
touch /home/ec2-user/.aws-ec2

# 4. Vérifier le .env
cat .env
# S'assurer que DATABASE_URL et REDIS_URL pointent vers harena-postgres et harena-redis

# 5. Installer Portainer (une seule fois)
docker volume create portainer_data
docker run -d \
  -p 9000:9000 \
  --name portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest

# 6. Configurer le monitoring automatique
crontab -e
# Ajouter: */5 * * * * /home/ec2-user/harena/healthcheck-monitor.sh

# 7. Déployer
chmod +x deploy-prod-final.sh
./deploy-prod-final.sh
```

### Déploiements suivants

```bash
# Connexion SSH
aws ssm start-session --region eu-west-1 --target i-0011b978b7cea66dc

# Déployer
cd /home/ec2-user/harena
./deploy-prod-final.sh
```

### Accès Prod

- **API publique**: http://63.35.52.216:3000-3008
- **PostgreSQL (DBeaver)**: 63.35.52.216:5432
  - Database: `harena`
  - User: `harena_admin`
  - Password: [depuis .env]
- **Portainer**: http://63.35.52.216:9000

---

## 📊 Monitoring

### Portainer (Dashboard)

**Dev**: http://localhost:9000
**Prod**: http://63.35.52.216:9000

Fonctionnalités:
- Vue d'ensemble des containers
- Logs en temps réel
- Ressources (CPU/RAM)
- Actions: restart, stop, rebuild

### Healthcheck Monitor (Prod uniquement)

```bash
# Voir les logs de monitoring
tail -f /var/log/harena-health.log

# Voir les alertes
cat /tmp/harena-alerts.txt

# Exécution manuelle
/home/ec2-user/harena/healthcheck-monitor.sh
```

### Logs Docker

```bash
# Dev
docker-compose -f docker-compose.dev.yml logs -f [service_name]

# Prod
docker-compose -f docker-compose.prod.yml logs -f [service_name]

# Logs d'un service spécifique
docker logs -f harena_conversation_v3
```

---

## 🔧 Commandes Utiles

### Dev

```bash
# État des services
docker-compose -f docker-compose.dev.yml ps

# Redémarrer un service
docker-compose -f docker-compose.dev.yml restart user_service

# Rebuild un service
docker-compose -f docker-compose.dev.yml up -d --build user_service

# Shell dans un conteneur
docker-compose -f docker-compose.dev.yml exec user_service bash

# Migrations
docker-compose -f docker-compose.dev.yml exec user_service alembic upgrade head
```

### Prod

```bash
# État des services
docker-compose -f docker-compose.prod.yml ps

# Healthchecks
docker ps --filter "health=healthy" | grep harena

# Vérifier les réseaux
docker network inspect harena-prod

# Ressources
docker stats --no-stream
free -h
df -h
```

---

## 🆘 Dépannage

### Service ne démarre pas

```bash
# Voir les logs
docker logs harena_[service_name]

# Rebuild from scratch
docker-compose -f docker-compose.[env].yml build --no-cache [service_name]
docker-compose -f docker-compose.[env].yml up -d [service_name]
```

### Erreur de connexion à PostgreSQL

**Dev**:
```bash
# Vérifier que PostgreSQL est UP
docker ps | grep postgres-dev

# Tester la connexion
docker exec postgres-dev pg_isready -U harena_dev
```

**Prod**:
```bash
# Vérifier le réseau
docker network inspect harena-prod | grep -E "harena-postgres|harena_user_service"

# Reconnecter si nécessaire
docker network connect harena-prod harena-postgres
```

### Port déjà utilisé

```bash
# Trouver le processus
netstat -ano | findstr :5174  # Windows
lsof -i :5174  # Linux/Mac

# Changer le port dans docker-compose.[env].yml
# Exemple: "5175:5173" au lieu de "5174:5173"
```

---

## 📚 Documentation Complète

- **Architecture**: `ARCHITECTURE_FINALE.md`
- **Monitoring**: `STABILITY_GUIDE.md`
- **Infrastructure**: `AWS_INFRASTRUCTURE.md`
- **Analyse**: `INFRASTRUCTURE_ANALYSIS.md`

---

**✅ Vous êtes prêt à développer et déployer!**
