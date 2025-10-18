# 📚 Documentation Backend Harena

Bienvenue dans la documentation des API backend Harena déployées sur AWS.

## 📖 Fichiers disponibles

### 🔍 Infrastructure Elasticsearch

- **[BONSAI_CONFIGURATION.md](./BONSAI_CONFIGURATION.md)**
  Configuration complète du cluster Bonsai en production (credentials, tests, monitoring)

- **[ELASTICSEARCH_MIGRATION_GUIDE.md](./ELASTICSEARCH_MIGRATION_GUIDE.md)**
  Guide complet de migration vers AWS ou autre service Elasticsearch

- **[ELASTICSEARCH_MAPPING_DIFFERENCES.md](../ELASTICSEARCH_MAPPING_DIFFERENCES.md)**
  Analyse comparative des mappings entre Bonsai et AWS

### 🚀 Pour commencer rapidement

- **[BACKEND_ENDPOINTS_QUICK_REFERENCE.md](./BACKEND_ENDPOINTS_QUICK_REFERENCE.md)**
  Référence ultra-rapide de tous les endpoints (1 page)

- **[backend-config.json](./backend-config.json)**
  Configuration JSON importable directement dans le frontend

### 📘 Documentation complète

- **[BACKEND_API_URLS.md](./BACKEND_API_URLS.md)**
  Documentation exhaustive de tous les services :
  - User Service (authentification, profils, métriques utilisateurs)
  - Conversation Service (IA conversationnelle, streaming, WebSocket)
  - Search Service (recherche de transactions, filtres)
  - Metric Service (métriques financières, tendances)
  - Enrichment Service (catégorisation, normalisation)
  - Exemples d'intégration frontend (React, JavaScript)

- **[BACKEND_USAGE_EXAMPLES.md](./BACKEND_USAGE_EXAMPLES.md)**
  Exemples de code complets et prêts à l'emploi :
  - Authentification (inscription, login, gestion du token)
  - Conversation (simple, streaming SSE, WebSocket)
  - Recherche de transactions
  - Métriques financières
  - Classe APIClient complète

## 🌐 Information de déploiement

- **IP Production**: `34.244.189.140`
- **Région AWS**: `eu-west-1` (Ireland)
- **Instance EC2**: `i-0cc135eda272700e1`
- **Environnement**: Production

## 🔐 Services disponibles

| Service | Port | Status | Documentation |
|---------|------|--------|---------------|
| User Service | 8000 | ✅ | [Swagger](http://34.244.189.140:8000/docs) |
| Conversation Service | 8001 | ✅ | [Swagger](http://34.244.189.140:8001/docs) |
| Enrichment Service | 8002 | ✅ | [Swagger](http://34.244.189.140:8002/docs) |
| Metric Service | 8004 | ✅ | [Swagger](http://34.244.189.140:8004/docs) |
| Search Service | 8005 | ✅ | [Swagger](http://34.244.189.140:8005/docs) |

## ⚡ Quick Start

### 1. Configuration du client API

```javascript
import config from './docs/backend-config.json';

const API = {
  user: config.services.user.baseURL,
  conversation: config.services.conversation.baseURL,
  search: config.services.search.baseURL,
  metrics: config.services.metrics.baseURL,
  enrichment: config.services.enrichment.baseURL
};
```

### 2. Login et obtention du token

```javascript
const response = await fetch(`${API.user}/api/v1/users/login`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password'
  })
});

const { access_token, user } = await response.json();
localStorage.setItem('auth_token', access_token);
```

### 3. Utiliser les services

```javascript
// Envoyer un message au chatbot
const token = localStorage.getItem('auth_token');

const chatResponse = await fetch(`${API.conversation}/api/v1/conversation/${user.id}`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    message: "Combien j'ai dépensé ce mois-ci?"
  })
});

const result = await chatResponse.json();
console.log(result.response.message);
```

## 📊 Fonctionnalités principales

### 🔐 User Service
- Authentification JWT
- Gestion des profils utilisateurs
- Métriques utilisateurs de base

### 💬 Conversation Service (⭐ Service phare)
- **IA conversationnelle** Phase 5 - Pipeline complet
- **Streaming SSE** pour réponses progressives
- **WebSocket** pour temps réel
- **Persistence** des conversations
- **JWT compatible** avec user_service
- **JSON Output enforced** pour réponses structurées

### 🔍 Search Service
- Recherche full-text de transactions
- Filtres multiples (catégorie, date, montant, marchand)
- Agrégations automatiques

### 📊 Metric Service
- Vue d'ensemble financière
- Répartition par catégorie
- Détection de dépenses récurrentes
- Tendances mensuelles
- Flux de trésorerie

### 🔄 Enrichment Service
- Catégorisation automatique par ML
- Normalisation des marchands
- Détection de duplicatas
- Enrichissement par API Bridge

## 🔑 Authentification

Tous les services (sauf auth endpoints) nécessitent un JWT token :

```javascript
headers: {
  'Authorization': 'Bearer YOUR_JWT_TOKEN'
}
```

Le token est obtenu via:
```
POST http://34.244.189.140:8000/api/v1/users/login
```

Durée de validité: **3600 secondes** (1 heure)

## 🛠️ Outils de développement

### Health Check de tous les services

```bash
#!/bin/bash
for port in 8000 8001 8002 8004 8005; do
  echo "Service on port $port:"
  curl -s http://34.244.189.140:$port/health | jq '.status'
done
```

### Test rapide d'authentification

```bash
# Login
curl -X POST http://34.244.189.140:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}'

# Récupérer le token et tester
TOKEN="YOUR_TOKEN_HERE"
curl http://34.244.189.140:8001/api/v1/conversation/1 \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Test"}'
```

## 📱 Intégration Frontend recommandée

1. **Utiliser la classe APIClient** fournie dans [BACKEND_USAGE_EXAMPLES.md](./BACKEND_USAGE_EXAMPLES.md)
2. **Importer la config** depuis `backend-config.json`
3. **Gérer le token** avec localStorage ou un state manager
4. **Implémenter le refresh** automatique du token avant expiration
5. **Gérer les erreurs 401** pour rediriger vers login

## 🔒 Sécurité

- ✅ JWT Authentication sur tous les endpoints sensibles
- ✅ CORS configuré pour origins de production
- ✅ Validation des entrées côté serveur
- ✅ Rate limiting (à venir)
- ⚠️ HTTPS recommandé pour production (actuellement HTTP)

## 📈 Monitoring

Tous les services exposent:
- `/health` - Health check standard
- `/metrics` - Métriques Prometheus
- `/docs` - Documentation Swagger UI

### Prometheus endpoints

```
http://34.244.189.140:8000/metrics
http://34.244.189.140:8001/metrics
http://34.244.189.140:8002/metrics
http://34.244.189.140:8004/metrics
http://34.244.189.140:8005/metrics
```

## 🐛 Debugging

### Voir les logs d'un service (depuis EC2)

```bash
# SSH ou SSM vers l'instance
aws ssm start-session --target i-0cc135eda272700e1 --region eu-west-1

# Voir les logs en temps réel
sudo journalctl -u conversation-service -f
sudo journalctl -u user-service -f
# etc.
```

### Tester un endpoint directement

```bash
# Test simple
curl http://34.244.189.140:8001/health

# Test avec JWT
curl -H "Authorization: Bearer TOKEN" \
  http://34.244.189.140:8001/api/v1/conversation/health
```

## 🔄 Versions

- **User Service**: 1.0.0
- **Conversation Service**: 2.0.0 (Phase 5)
- **Search Service**: 1.0.0
- **Metric Service**: 1.0.0
- **Enrichment Service**: 1.0.0

## 📞 Support

Pour toute question ou problème:
1. Consulter la documentation complète dans ce dossier
2. Vérifier les logs sur l'instance EC2
3. Tester les endpoints avec `/docs` (Swagger UI)

## 🎯 Roadmap

- [ ] Migration vers HTTPS (avec reverse proxy Nginx)
- [ ] Rate limiting par utilisateur
- [ ] Cache Redis pour métriques
- [ ] WebSocket authentification améliorée
- [ ] API Gateway unifié
- [ ] Monitoring centralisé (Grafana + Prometheus)

---

**Dernière mise à jour**: 2025-10-08
**Équipe**: Harena Backend Team
