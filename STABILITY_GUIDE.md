# 🛡️ Guide de Stabilité Infrastructure Harena

## 📋 État actuel de l'infrastructure

**Instance:** i-0011b978b7cea66dc (harena-allinone-dev)
**Uptime:** 7+ jours
**Services:** 7 conteneurs (tous HEALTHY)
**Version:** v6.0.4

---

## ⚠️ Risques identifiés et solutions

### 1. **Problème de configuration de ports (résolu)**

**Symptôme:** Services inaccessibles de l'extérieur (127.0.0.1 au lieu de 0.0.0.0)

**Cause:** Conteneurs reconstruits avec une ancienne configuration

**Solution permanente:**
- ✅ Toujours faire `docker-compose down && build && up` complet
- ✅ Utiliser le script `deploy-verified.sh` qui vérifie les ports
- ✅ Monitoring automatique avec `healthcheck-monitor.sh`

---

## 🚀 Procédures de déploiement sécurisées

### Déploiement standard (recommandé)

```bash
# Sur l'instance EC2
cd /home/ec2-user/harena
./deploy-verified.sh
```

Ce script :
1. Pull le code depuis git
2. Arrête proprement les services
3. Rebuild TOUTES les images (--no-cache)
4. Démarre les services
5. **Vérifie automatiquement** :
   - Conteneurs UP
   - Healthchecks HEALTHY
   - Ports sur 0.0.0.0
   - Endpoints HTTP accessibles
   - Ressources système

### Déploiement d'urgence (service spécifique)

```bash
# Exemple: redéployer uniquement conversation_service_v3
cd /home/ec2-user/harena
docker-compose -f docker-compose.prod.yml build conversation_service_v3
docker-compose -f docker-compose.prod.yml up -d conversation_service_v3

# IMPORTANT: Vérifier après
docker ps --filter name=conversation_v3
curl http://localhost:3008/health
```

---

## 📊 Monitoring continu

### Configuration du monitoring automatique

1. **Installer le healthcheck en cron**

```bash
# Sur l'instance EC2
sudo crontab -e

# Ajouter cette ligne (vérification toutes les 5 minutes)
*/5 * * * * /home/ec2-user/harena/healthcheck-monitor.sh

# Vérification plus fréquente (toutes les minutes) pour prod critique
* * * * * /home/ec2-user/harena/healthcheck-monitor.sh
```

2. **Consulter les logs de monitoring**

```bash
# Voir les derniers healthchecks
tail -50 /var/log/harena-health.log

# Voir uniquement les alertes
cat /tmp/harena-alerts.txt

# Suivre en temps réel
tail -f /var/log/harena-health.log
```

### Métriques à surveiller

| Métrique | Seuil normal | Alerte si |
|----------|--------------|-----------|
| Uptime conteneurs | 100% | < 95% |
| Healthchecks | All HEALTHY | Any UNHEALTHY |
| Mémoire système | < 85% | > 90% |
| Disque système | < 85% | > 90% |
| Erreurs/heure | < 50 | > 100 |
| Response time API | < 2s | > 5s |

---

## 🔧 Commandes de diagnostic rapide

### Vérifier l'état global

```bash
# Statut de tous les services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Healthchecks
docker ps --filter "health=unhealthy"

# Ressources système
free -h
df -h
uptime
```

### Vérifier un service spécifique

```bash
# Logs en temps réel
docker logs -f harena_conversation_v3_prod

# Dernières erreurs
docker logs harena_conversation_v3_prod 2>&1 | grep -i error | tail -20

# Inspect configuration
docker inspect harena_conversation_v3_prod | grep -A 10 "PortBindings"
```

### Tests des endpoints

```bash
# Health checks
curl http://localhost:3000/health
curl http://localhost:3001/api/v1/search/health
curl http://localhost:3002/health
curl http://localhost:3006/health
curl http://localhost:3008/health

# Test complet d'authentification
curl -X POST http://localhost:3000/api/v1/users/auth/login \
  -H 'content-type: multipart/form-data' \
  -F username=henri@example.com \
  -F password=Henri123456
```

---

## 🚨 Procédures d'urgence

### Service down ou unhealthy

```bash
# 1. Identifier le service
docker ps -a | grep -v "Up.*healthy"

# 2. Voir les logs d'erreur
docker logs [service_name] --tail 100

# 3. Redémarrer le service
docker-compose -f docker-compose.prod.yml restart [service_name]

# 4. Si échec: rebuild
docker-compose -f docker-compose.prod.yml build [service_name]
docker-compose -f docker-compose.prod.yml up -d [service_name]

# 5. Vérifier
docker ps --filter name=[service_name]
curl http://localhost:[port]/health
```

### Mémoire saturée

```bash
# 1. Identifier les consommateurs
docker stats --no-stream

# 2. Nettoyer les ressources Docker
docker system prune -f

# 3. Redémarrer les services gourmands
docker-compose -f docker-compose.prod.yml restart

# 4. Si critique: redémarrer l'instance EC2 (dernier recours)
sudo reboot
```

### Disque saturé

```bash
# 1. Identifier la consommation
du -sh /home/ec2-user/* | sort -h

# 2. Nettoyer les logs Docker
docker system prune -a -f --volumes

# 3. Nettoyer les anciennes images
docker image prune -a -f

# 4. Vérifier l'espace libéré
df -h
```

---

## 📈 Améliorations recommandées

### Court terme (1-2 semaines)

- [ ] **Mettre en place le monitoring automatique** (cron + healthcheck-monitor.sh)
- [ ] **Configurer des alertes email** en cas de problème
- [ ] **Documenter les procédures de rollback**
- [ ] **Tester un redéploiement complet** pour valider les procédures

### Moyen terme (1 mois)

- [ ] **CloudWatch monitoring** pour EC2 (CPU, mémoire, disque)
- [ ] **CloudWatch Logs** pour centraliser les logs des conteneurs
- [ ] **SNS alerting** pour notifications critiques
- [ ] **Backup automatique** de la base PostgreSQL
- [ ] **Auto-scaling** si charge augmente

### Long terme (3+ mois)

- [ ] **ECS/Fargate** pour orchestration native AWS
- [ ] **RDS** pour PostgreSQL managé (haute disponibilité)
- [ ] **ElastiCache** pour Redis managé
- [ ] **ALB** avec target groups pour load balancing
- [ ] **CI/CD pipeline** (GitHub Actions → ECR → ECS)
- [ ] **Blue/Green deployments** pour zero-downtime

---

## 🎯 Checklist quotidienne (2 minutes)

```bash
# 1. Vérifier que tous les services sont UP
docker ps | grep -c "healthy"  # Doit afficher 7

# 2. Pas d'alertes
cat /tmp/harena-alerts.txt  # Doit être vide

# 3. Ressources OK
free -h && df -h  # Mémoire et disque < 85%

# 4. Logs propres (pas d'erreurs massives)
docker-compose -f /home/ec2-user/harena/docker-compose.prod.yml logs --since 1h | grep -i error | wc -l
# Doit être < 50
```

---

## 📞 Support et escalade

### Niveaux d'alerte

- **🟢 Normal:** Tous les checks passent → Aucune action
- **🟡 Warning:** Ressources > 85% → Surveiller de près
- **🟠 Dégradé:** 1 service unhealthy → Investiguer sous 30min
- **🔴 Critique:** Multiple services down → Action immédiate

### Contacts

- **Développeur:** [votre_email]
- **AWS Support:** Via console AWS (Basic/Developer/Business plan)
- **Documentation:** Ce fichier + AWS_INFRASTRUCTURE.md

---

## 📝 Historique des incidents

### 2025-10-24: Port binding incorrect

**Problème:** user_service et budget_service inaccessibles (127.0.0.1)
**Impact:** Impossible de générer des tokens JWT
**Cause:** Conteneurs non rebuilté après mise à jour du code
**Solution:** Redéploiement complet avec vérification
**Prévention:** Script deploy-verified.sh + monitoring automatique

---

**Dernière mise à jour:** 2025-10-24
**Version du guide:** 1.0
**Mainteneur:** Équipe DevOps Harena
