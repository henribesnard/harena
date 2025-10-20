# Guide de Déploiement - Conversation Service v3

## 🚀 Configuration

### Port et Préfixe API
- **Port**: 3008
- **Préfixe API**: `/api/v3`
- **Endpoints**:
  - `http://localhost:3008/health`
  - `http://localhost:3008/api/v3/conversation/ask`
  - `http://localhost:3008/api/v3/conversation/stats`
  - `http://localhost:3008/api/v3/conversation/health`

### Service Search
Le service utilise `metric_service` (port 3002) pour les requêtes Elasticsearch.

---

## 📦 Déploiement Docker (Recommandé)

### 1. Build et Démarrage

```bash
# Depuis la racine du projet harena
docker-compose up -d conversation_service_v3
```

### 2. Vérification

```bash
# Vérifier les logs
docker logs harena_conversation_v3 --tail 50

# Suivre les logs en temps réel
docker logs -f harena_conversation_v3

# Vérifier le status
docker ps | grep conversation_v3
```

### 3. Test

```bash
# Health check
curl http://localhost:3008/health

# Test conversation
curl -X POST http://localhost:3008/api/v3/conversation/ask \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "message": "Combien j'\''ai dépensé en courses ce mois-ci ?"
  }'
```

### 4. Commandes Utiles

```bash
# Redémarrer
docker-compose restart conversation_service_v3

# Rebuild (après changement de code)
docker-compose up -d --build conversation_service_v3

# Arrêter
docker-compose stop conversation_service_v3

# Supprimer
docker-compose down conversation_service_v3
```

---

## 💻 Déploiement Local (Développement)

### 1. Installation

```bash
# Aller dans le dossier
cd conversation_service_v3

# Créer environnement virtuel
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/Mac)
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copier le template
cp .env.example .env

# Éditer .env et ajouter votre OPENAI_API_KEY
notepad .env  # Windows
nano .env     # Linux/Mac
```

Variables critiques:
```bash
OPENAI_API_KEY=sk-...                    # REQUIS
SEARCH_SERVICE_URL=http://localhost:3002  # URL du metric_service
PORT=3008
```

### 3. Démarrage

```bash
uvicorn app.main:app --reload --port 3008
```

### 4. Test

```bash
python test_service.py
```

---

## 🌐 Configuration Docker Compose

Le service est déjà configuré dans `docker-compose.yml`:

```yaml
conversation_service_v3:
  build:
    context: ./conversation_service_v3
    dockerfile: Dockerfile
  container_name: harena_conversation_v3
  ports:
    - "3008:3008"
  environment:
    - SEARCH_SERVICE_URL=http://harena_metric_service:3002
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - LLM_MODEL=gpt-4o-mini
    - LLM_RESPONSE_MODEL=gpt-4o
    - PORT=3008
    - API_V3_PREFIX=/api/v3
  depends_on:
    - metric_service
```

### Variables d'Environnement

Assurez-vous d'avoir dans votre `.env` racine:

```bash
# OpenAI (REQUIS pour v3)
OPENAI_API_KEY=sk-...

# Configuration LLM (optionnel, valeurs par défaut)
LLM_MODEL=gpt-4o-mini
LLM_RESPONSE_MODEL=gpt-4o
LLM_TEMPERATURE=0.1

# Agent Configuration (optionnel)
MAX_CORRECTION_ATTEMPTS=2
QUERY_TIMEOUT_SECONDS=30

# Logging (optionnel)
LOG_LEVEL=INFO
```

---

## 🔄 Migration depuis v0/v2

### Compatibilité API

Les trois versions peuvent coexister:
- **v0**: `http://localhost:3003/api/v1/conversation/...`
- **v2**: `http://localhost:3007/api/v2/conversation/...`
- **v3**: `http://localhost:3008/api/v3/conversation/...`

### Migration Progressive

1. **Déployer v3** en parallèle de v2
2. **Router 10%** du trafic vers v3
3. **Monitorer** les performances et erreurs
4. **Augmenter** progressivement le trafic
5. **Désactiver** v2 quand v3 est stable

### Changements API

**Format de Requête** (identique):
```json
{
  "user_id": 1,
  "message": "Question utilisateur",
  "conversation_id": "optional",
  "context": []
}
```

**Format de Réponse** (légèrement différent):
```json
{
  "success": true,
  "message": "Réponse formatée...",
  "total_results": 12,
  "aggregations_summary": "...",
  "metadata": {
    "pipeline_time_ms": 2100,
    "query_analysis": {...}
  }
}
```

---

## 📊 Monitoring

### Endpoints de Monitoring

```bash
# Health check basique
curl http://localhost:3008/health

# Health check détaillé (agents)
curl http://localhost:3008/api/v3/conversation/health

# Statistiques complètes
curl http://localhost:3008/api/v3/conversation/stats
```

### Logs Docker

```bash
# Logs en temps réel
docker logs -f harena_conversation_v3

# Dernières 100 lignes
docker logs harena_conversation_v3 --tail 100

# Logs avec timestamps
docker logs -t harena_conversation_v3
```

### Métriques Importantes

Vérifier dans `/stats`:
- `success_rate`: Doit être > 90%
- `correction_rate`: Normal entre 10-20%
- `avg_pipeline_time_ms`: Doit être < 3000ms

---

## 🐛 Troubleshooting

### Service ne démarre pas

```bash
# Vérifier les logs
docker logs harena_conversation_v3

# Vérifier les dépendances
docker-compose ps | grep metric_service

# Rebuild complet
docker-compose down conversation_service_v3
docker-compose up -d --build conversation_service_v3
```

### Erreur "OpenAI API key not found"

```bash
# Vérifier que la clé est dans .env
grep OPENAI_API_KEY .env

# Redémarrer le service
docker-compose restart conversation_service_v3
```

### Erreur "Search service unavailable"

```bash
# Vérifier que metric_service est up
docker ps | grep metric_service

# Tester metric_service
curl http://localhost:3002/health

# Vérifier la config réseau
docker inspect harena_conversation_v3 | grep -A 10 "Networks"
```

### Timeout sur les requêtes

```bash
# Augmenter le timeout dans docker-compose.yml
environment:
  - QUERY_TIMEOUT_SECONDS=60

# Redémarrer
docker-compose restart conversation_service_v3
```

### Réponses de mauvaise qualité

```bash
# Changer le modèle LLM
environment:
  - LLM_MODEL=gpt-4o  # Plus lent mais meilleur
  - LLM_RESPONSE_MODEL=gpt-4o

# Redémarrer
docker-compose restart conversation_service_v3
```

---

## 🔒 Sécurité

### JWT Token

Le service supporte l'authentification JWT:

```bash
curl -X POST http://localhost:3008/api/v3/conversation/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{...}'
```

### Rate Limiting

À implémenter selon les besoins:
- Nginx reverse proxy avec rate limiting
- Redis-based rate limiter dans le code
- API Gateway avec quotas

---

## 📈 Performance

### Optimisations Production

```yaml
# Dans docker-compose.yml
conversation_service_v3:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
  environment:
    - LOG_LEVEL=WARNING  # Moins de logs
    - MAX_CORRECTION_ATTEMPTS=1  # Moins de retries
```

### Scaling Horizontal

```bash
# Lancer plusieurs instances
docker-compose up -d --scale conversation_service_v3=3

# Ajouter un load balancer (nginx)
# Configurer round-robin sur port 3008
```

---

## 🔄 Mise à Jour

### Update du Code

```bash
# Pull les changements
git pull origin main

# Rebuild et redémarrer
docker-compose up -d --build conversation_service_v3

# Vérifier les logs
docker logs harena_conversation_v3 --tail 50
```

### Update des Dépendances

```bash
# Éditer requirements.txt
# Rebuild l'image
docker-compose build --no-cache conversation_service_v3
docker-compose up -d conversation_service_v3
```

---

## ✅ Checklist de Déploiement

### Pré-déploiement
- [ ] OPENAI_API_KEY configurée dans .env
- [ ] metric_service (port 3002) accessible
- [ ] Docker et docker-compose installés
- [ ] Port 3008 disponible

### Déploiement
- [ ] Build: `docker-compose build conversation_service_v3`
- [ ] Start: `docker-compose up -d conversation_service_v3`
- [ ] Check logs: Pas d'erreur au démarrage
- [ ] Health check: `curl http://localhost:3008/health`

### Post-déploiement
- [ ] Test conversation: Request réussie
- [ ] Vérifier stats: Success rate > 90%
- [ ] Monitorer logs pendant 1h
- [ ] Tester auto-correction avec query invalide
- [ ] Vérifier latence < 3s

---

## 📞 Support

### Logs Complets pour Debug

```bash
# Activer DEBUG logging
docker-compose stop conversation_service_v3

# Éditer docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG

# Redémarrer
docker-compose up -d conversation_service_v3

# Suivre les logs
docker logs -f harena_conversation_v3
```

### Informations à Collecter

Si problème:
1. Logs: `docker logs harena_conversation_v3 > logs.txt`
2. Stats: `curl http://localhost:3008/api/v3/conversation/stats > stats.json`
3. Health: `curl http://localhost:3008/api/v3/conversation/health > health.json`
4. Exemple de requête qui échoue

---

## 🎯 Résumé Commandes Essentielles

```bash
# Démarrage
docker-compose up -d conversation_service_v3

# Logs
docker logs -f harena_conversation_v3

# Test
curl http://localhost:3008/health

# Stats
curl http://localhost:3008/api/v3/conversation/stats

# Redémarrage
docker-compose restart conversation_service_v3

# Rebuild
docker-compose up -d --build conversation_service_v3
```

---

Le service v3 est maintenant déployé et prêt à recevoir des requêtes sur `http://localhost:3008/api/v3/` ! 🚀
