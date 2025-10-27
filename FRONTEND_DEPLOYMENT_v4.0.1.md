# Déploiement Frontend v4.0.1 sur AWS

## Vue d'ensemble

Ce document détaille le processus de déploiement de la version v4.0.1 du frontend Harena sur AWS EC2.

## Changements de la version v4.0.1

### Améliorations de l'interface
- **Sidebar unifiée** : La sidebar de conversation (historique des messages) est désormais fixe sur toutes les pages
- **Suppression de la navigation redondante** : Retrait de l'ancienne sidebar de navigation
- **Dashboard utilisateur amélioré** :
  - Suppression de la carte "Informations du profil" redondante
  - Déplacement de la section "Statistiques clés" après les métriques financières
  - Mise en page horizontale du "Résumé budgétaire" pour une meilleure lisibilité
- **Correction de bug** : Résolution du problème de page blanche sur `/configuration` (useState → useEffect)

### Modifications techniques
- Refactorisation de `ConversationSidebar.tsx` pour rendre les props optionnels
- Consolidation de tous les routes sous `MainLayout`
- Suppression de `ChatLayout` devenu obsolète
- Amélioration de la navigation mobile

## Prérequis

### 1. Configuration AWS
- **Instance EC2** : `i-0011b978b7cea66dc` (t3.small)
- **IP publique** : `63.35.52.216`
- **AWS CLI** configuré avec les bonnes permissions SSM
- **Accès SSM** à l'instance EC2

### 2. Vérifications avant déploiement
```bash
# Vérifier la connexion AWS
aws sts get-caller-identity

# Vérifier l'accès SSM à l'instance
aws ssm describe-instance-information \
  --instance-ids i-0011b978b7cea66dc
```

### 3. État des repositories Git

#### Backend (harena/)
- Tag actuel : `v6.1.2`
- Modifications à commit :
  - `docker-compose.aws.yml` (correction du port frontend 8080:80)
  - `update-frontend-aws.sh` (nouveau script de déploiement)

#### Frontend (harena_front/)
- Tag : `v4.0.1` ✅ (déjà créé et pushé)
- Dépôt séparé avec son propre remote Git

## Modifications de configuration

### docker-compose.aws.yml

**Corrections effectuées** :
```yaml
frontend:
  build:
    context: ./harena_front
    dockerfile: Dockerfile  # Utilise Nginx pour la production
    args:
      VITE_USER_SERVICE_URL: /api/v1
      VITE_SEARCH_SERVICE_URL: /api/v1
      VITE_METRIC_SERVICE_URL: /api/v1
      VITE_CONVERSATION_SERVICE_URL: /api/v3
      VITE_BUDGET_PROFILING_API_URL: /api/v1/budget  # ✅ Corrigé
  ports:
    - "8080:80"  # ✅ Corrigé (était 5173:5173)
  healthcheck:
    test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost"]  # ✅ Port 80
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s
```

**Explications** :
- Le Dockerfile du frontend utilise une build multi-stage : Node (builder) + Nginx (production)
- Le frontend sert les assets via Nginx sur le port 80 (standard HTTP)
- Le port externe 8080 permet d'accéder au frontend sans conflit avec le Nginx reverse proxy principal

### nginx/conf.d/harena.conf

**Vérification effectuée** : Configuration correcte ✅

```nginx
# Le proxy_pass pointe correctement vers le service frontend
location / {
    proxy_pass http://frontend;  # Résout vers frontend:80 par défaut
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    # ... autres headers
}
```

## Processus de déploiement

### Méthode 1 : Script automatisé (Recommandé)

#### Étape 1 : Commit des changements backend
```bash
cd ~/Projets/harena

# Ajouter les fichiers modifiés
git add docker-compose.aws.yml update-frontend-aws.sh FRONTEND_DEPLOYMENT_v4.0.1.md

# Commit avec message descriptif
git commit -m "feat(deployment): Add frontend v4.0.1 deployment configuration and script

- Fix docker-compose.aws.yml frontend port mapping (8080:80)
- Fix healthcheck to use port 80 instead of 5173
- Correct VITE_BUDGET_PROFILING_API_URL to /api/v1/budget
- Add update-frontend-aws.sh script for automated frontend deployment
- Add FRONTEND_DEPLOYMENT_v4.0.1.md documentation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Tag et push
git tag v6.1.3
git push origin main
git push origin v6.1.3
```

#### Étape 2 : Exécuter le script de déploiement
```bash
cd ~/Projets/harena

# Rendre le script exécutable
chmod +x update-frontend-aws.sh

# Lancer le déploiement
./update-frontend-aws.sh
```

Le script effectue automatiquement :
1. ✅ Vérification de l'état actuel du frontend
2. ✅ Pull du code frontend depuis Git (tag v4.0.1)
3. ✅ Mise à jour de docker-compose.aws.yml
4. ✅ Build du nouveau frontend (3-5 minutes)
5. ✅ Arrêt de l'ancien frontend
6. ✅ Démarrage du nouveau frontend
7. ✅ Tests de vérification

### Méthode 2 : Déploiement manuel via SSH

Si vous préférez contrôler chaque étape :

```bash
# Connexion SSH
export EC2_IP=63.35.52.216
export EC2_USER=ec2-user
export SSH_KEY=~/.ssh/harena-aws.pem

ssh -i $SSH_KEY $EC2_USER@$EC2_IP

# Une fois connecté sur l'EC2
cd ~/harena

# 1. Pull du backend (docker-compose.aws.yml mis à jour)
git pull origin main
git checkout v6.1.3

# 2. Pull du frontend
cd harena_front
git fetch --tags
git checkout v4.0.1
git pull origin main
cd ..

# 3. Build du frontend
docker-compose -f docker-compose.aws.yml build --no-cache frontend

# 4. Redémarrage du frontend
docker-compose -f docker-compose.aws.yml stop frontend
docker-compose -f docker-compose.aws.yml rm -f frontend
docker-compose -f docker-compose.aws.yml up -d frontend

# 5. Vérification
docker-compose -f docker-compose.aws.yml ps frontend
docker-compose -f docker-compose.aws.yml logs -f frontend
```

## Vérification post-déploiement

### 1. Vérifier l'état des containers

Via SSM :
```bash
aws ssm send-command \
  --instance-ids i-0011b978b7cea66dc \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd ~/harena && docker-compose -f docker-compose.aws.yml ps"]' \
  --output text
```

Via SSH :
```bash
ssh -i ~/.ssh/harena-aws.pem ec2-user@63.35.52.216
cd ~/harena
docker-compose -f docker-compose.aws.yml ps
```

**Résultat attendu** :
```
NAME                                 STATUS          PORTS
harena-frontend-1                    Up (healthy)    0.0.0.0:8080->80/tcp
```

### 2. Tester l'accès au frontend

#### Test depuis votre machine locale
```bash
# Test du health check du reverse proxy Nginx
curl http://63.35.52.216/health
# Attendu : OK

# Test de la page d'accueil du frontend
curl -I http://63.35.52.216:8080/
# Attendu : HTTP/1.1 200 OK

# Test via le reverse proxy (accès normal)
curl -I http://63.35.52.216/
# Attendu : HTTP/1.1 200 OK
```

#### Test dans le navigateur
- **Frontend direct** : http://63.35.52.216:8080
- **Via reverse proxy** : http://63.35.52.216
- **Grafana** : http://63.35.52.216:3033

### 3. Vérifier les logs

```bash
# Logs du frontend
docker-compose -f docker-compose.aws.yml logs -f frontend

# Logs du reverse proxy Nginx
docker-compose -f docker-compose.aws.yml logs -f nginx
```

**Logs normaux attendus** :
```
frontend-1  | /docker-entrypoint.sh: Launching /docker-entrypoint.d/40-envsubst-on-templates.sh
frontend-1  | /docker-entrypoint.sh: Configuration complete; ready for start up
```

### 4. Tests fonctionnels

Une fois le frontend accessible, tester :

✅ **Pages principales** :
- [ ] Login / Register
- [ ] Chat (avec sidebar de conversation)
- [ ] Dashboard utilisateur
- [ ] Budget Profiling
- [ ] Configuration

✅ **Nouvelle interface** :
- [ ] La sidebar de conversation est visible sur toutes les pages
- [ ] Le menu mobile fonctionne (bouton hamburger)
- [ ] Le dashboard affiche le layout horizontal
- [ ] La page Configuration se charge sans erreur

✅ **APIs** :
- [ ] Connexion utilisateur fonctionne
- [ ] Récupération des métriques financières
- [ ] Chat avec l'assistant IA
- [ ] Sauvegarde des préférences utilisateur

## Monitoring

### Grafana
- **URL** : http://63.35.52.216:3033
- **Identifiants** : admin / HarenaAdmin2024!

**Dashboards à surveiller** :
- Docker Container Metrics
- Nginx Access Logs
- Application Performance

### Logs centralisés (Loki)
```bash
# Via CLI (sur l'EC2)
docker-compose -f docker-compose.aws.yml logs -f frontend | grep -i error
```

## Rollback en cas de problème

Si le déploiement échoue, revenir à la version précédente :

```bash
cd ~/harena/harena_front

# 1. Revenir au tag précédent (à adapter selon votre historique)
git checkout v4.0.0

# 2. Rebuild
cd ~/harena
docker-compose -f docker-compose.aws.yml build --no-cache frontend

# 3. Redémarrer
docker-compose -f docker-compose.aws.yml up -d frontend

# 4. Vérifier
docker-compose -f docker-compose.aws.yml ps frontend
docker-compose -f docker-compose.aws.yml logs frontend
```

## Troubleshooting

### Problème 1 : Le frontend ne démarre pas

**Symptômes** :
```
frontend-1  | exited with code 1
```

**Solutions** :
```bash
# Vérifier les logs détaillés
docker-compose -f docker-compose.aws.yml logs frontend

# Vérifier que le build s'est bien passé
docker images | grep frontend

# Rebuild complet
docker-compose -f docker-compose.aws.yml build --no-cache frontend
docker-compose -f docker-compose.aws.yml up -d frontend
```

### Problème 2 : Page blanche dans le navigateur

**Symptômes** :
- Le container est UP mais la page ne se charge pas
- Console browser : erreurs de connexion aux APIs

**Solutions** :
```bash
# 1. Vérifier les variables d'environnement du build
docker-compose -f docker-compose.aws.yml config | grep -A 10 frontend

# 2. Vérifier le fichier nginx.conf du frontend
docker-compose -f docker-compose.aws.yml exec frontend cat /etc/nginx/conf.d/default.conf

# 3. Tester les APIs backend
curl http://localhost/api/v1/health
curl http://localhost/api/v3/health
```

### Problème 3 : Erreur 502 Bad Gateway

**Symptômes** :
```
502 Bad Gateway
nginx/1.24.0
```

**Solutions** :
```bash
# 1. Vérifier que le frontend écoute sur le bon port
docker-compose -f docker-compose.aws.yml exec frontend netstat -tlnp

# 2. Vérifier la configuration du reverse proxy
docker-compose -f docker-compose.aws.yml exec nginx nginx -t

# 3. Redémarrer le reverse proxy
docker-compose -f docker-compose.aws.yml restart nginx
```

### Problème 4 : Mémoire insuffisante

**Symptômes** :
```
Build failed: Cannot allocate memory
```

**Solutions** :
```bash
# Vérifier l'utilisation mémoire
free -h

# Nettoyer les images inutilisées
docker system prune -a

# Augmenter le swap si nécessaire (déjà configuré à 2GB)
sudo swapon -s
```

## Architecture du déploiement

```
┌─────────────────────────────────────────────────────────────┐
│                         Internet                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   EC2 Instance   │
              │  63.35.52.216    │
              └──────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌─────────────────┐         ┌──────────────────┐
│  Port 80        │         │  Port 8080       │
│  Nginx Reverse  │────────▶│  Frontend        │
│  Proxy          │         │  (Nginx + React) │
└─────────────────┘         └──────────────────┘
          │
          │ Proxy vers...
          │
          ├─────▶ /api/v1/users → user_service:3000
          ├─────▶ /api/v1/search → search_service:3001
          ├─────▶ /api/v1/metrics → metric_service:3002
          ├─────▶ /api/v3/ → conversation_service:3005
          └─────▶ /api/v1/budget → budget_profiling_service:3006
```

## Checklist de déploiement

### Avant le déploiement
- [ ] Code frontend committé et tagué (v4.0.1) ✅
- [ ] Configuration docker-compose.aws.yml mise à jour ✅
- [ ] Script de déploiement créé ✅
- [ ] Documentation à jour ✅
- [ ] AWS CLI configuré
- [ ] Accès SSM vérifié

### Pendant le déploiement
- [ ] Backup de l'état actuel (optionnel)
- [ ] Exécution du script update-frontend-aws.sh
- [ ] Surveillance des logs en temps réel
- [ ] Temps de build : ~3-5 minutes

### Après le déploiement
- [ ] Container frontend UP et healthy
- [ ] Frontend accessible sur le port 8080
- [ ] Frontend accessible via reverse proxy (port 80)
- [ ] Tests fonctionnels passés
- [ ] Monitoring Grafana vérifié
- [ ] Pas d'erreur dans les logs

## Points d'attention

### 1. Repositories séparés
- **Backend** : `~/Projets/harena/` - Contient docker-compose.aws.yml et scripts de déploiement
- **Frontend** : `~/Projets/harena/harena_front/` - Sous-dossier avec son propre dépôt Git

Sur l'EC2, la structure est :
```
~/harena/
├── docker-compose.aws.yml
├── harena_front/  (clone Git séparé)
│   ├── .git/
│   ├── src/
│   ├── Dockerfile
│   └── ...
├── user_service/
├── conversation_service_v3/
└── ...
```

### 2. Configuration Nginx du frontend

Le frontend utilise un Dockerfile multi-stage :
1. **Stage 1** : Build avec Node.js et Vite
2. **Stage 2** : Serve avec Nginx

Le fichier `nginx.conf` du frontend doit rediriger toutes les routes vers `index.html` pour le routing React :
```nginx
location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

### 3. Variables d'environnement

Les variables `VITE_*` sont injectées au moment du **build** (pas au runtime) :
- `VITE_USER_SERVICE_URL=/api/v1`
- `VITE_SEARCH_SERVICE_URL=/api/v1`
- `VITE_METRIC_SERVICE_URL=/api/v1`
- `VITE_CONVERSATION_SERVICE_URL=/api/v3`
- `VITE_BUDGET_PROFILING_API_URL=/api/v1/budget`

Ces URLs relatives permettent au frontend de communiquer avec les APIs via le reverse proxy Nginx.

## Support et contacts

- **Documentation principale** : `DEPLOYMENT.md`
- **Infrastructure AWS** : `AWS_INFRASTRUCTURE.md`
- **Monitoring** : Grafana @ http://63.35.52.216:3033

---

**Version** : 1.0.0
**Date** : 2025-10-27
**Auteur** : Claude Code
**Frontend Version** : v4.0.1
