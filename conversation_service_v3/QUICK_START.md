# Quick Start Guide - Conversation Service V3

## 🚀 Démarrage Rapide

### 1. Prérequis

```bash
# Python 3.10+
python --version

# Dependencies
cd conversation_service_v3
pip install -r requirements.txt
```

### 2. Configuration

Créez un fichier `.env` :

```bash
# Copier l'exemple
cp .env.example .env

# Éditer avec vos clés
nano .env
```

Contenu minimum du `.env` :

```env
# OpenAI API Key (requis)
OPENAI_API_KEY=sk-your-openai-key-here

# LLM Configuration
LLM_MODEL=gpt-4o-mini
LLM_RESPONSE_MODEL=gpt-4o
LLM_TEMPERATURE=0.1

# Services
SEARCH_SERVICE_URL=http://localhost:3002

# Agent Configuration
MAX_CORRECTION_ATTEMPTS=2
QUERY_TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
```

### 3. Démarrer le Service

```bash
# Depuis le dossier conversation_service_v3
python -m uvicorn app.main:app --host 0.0.0.0 --port 3008 --reload
```

Vous devriez voir :

```
INFO:     Uvicorn running on http://0.0.0.0:3008 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 4. Vérifier que ça marche

Ouvrez un nouveau terminal :

```bash
# Health check
curl http://localhost:3008/health

# Devrait retourner:
# {"status":"healthy","service":"conversation_service_v3",...}
```

### 5. Tester l'API

#### Option A : Avec curl

```bash
curl -X POST http://localhost:3008/api/v1/conversation/3 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "client_info": {
      "platform": "web",
      "version": "1.0.0"
    },
    "message": "Mes dépenses de plus de 100 euros",
    "message_type": "text",
    "priority": "normal"
  }'
```

#### Option B : Avec le script de test

```bash
# Test simple
python test_v1_compatibility.py

# Test détaillé avec analyse
python test_detailed.py
```

#### Option C : Avec Python

```python
import requests

url = "http://localhost:3008/api/v1/conversation/3"

payload = {
    "client_info": {
        "platform": "web",
        "version": "1.0.0"
    },
    "message": "Mes dépenses de plus de 100 euros",
    "message_type": "text",
    "priority": "normal"
}

headers = {
    "authorization": "Bearer your-jwt-token",
    "content-type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## 📝 Tests Détaillés

### Lancer les tests complets

```bash
python test_detailed.py
```

Ce script va :
- ✅ Vérifier que le service est accessible
- ✅ Tester plusieurs types de requêtes
- ✅ Mesurer les temps de réponse
- ✅ Vérifier la structure des réponses
- ✅ Identifier les bugs automatiquement
- ✅ Sauvegarder les résultats dans un fichier JSON

### Analyser les résultats

Les résultats sont sauvegardés dans `test_results_YYYYMMDD_HHMMSS.json`

```bash
# Voir le dernier résultat
cat test_results_*.json | jq .
```

## 🐛 Troubleshooting

### Le service ne démarre pas

**Erreur : "OPENAI_API_KEY not found"**
```bash
# Vérifier le .env
cat .env | grep OPENAI_API_KEY

# Doit contenir :
# OPENAI_API_KEY=sk-...
```

**Erreur : "Port already in use"**
```bash
# Trouver le processus qui utilise le port 3008
# Windows
netstat -ano | findstr :3008

# Linux/Mac
lsof -i :3008

# Tuer le processus ou changer le port
python -m uvicorn app.main:app --port 3009
```

**Erreur : "Module not found"**
```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Les tests échouent

**"Connection Error"**
```bash
# Vérifier que le service tourne
curl http://localhost:3008/health
```

**"Timeout"**
- Le LLM peut prendre du temps à répondre (normal)
- Vérifiez votre connexion internet
- Vérifiez que search_service est accessible

**"Invalid JWT"**
- Le token JWT doit être valide
- Pour les tests, vous pouvez temporairement désactiver l'auth
- Ou utiliser un token de test valide

### Logs et Debug

```bash
# Voir les logs en temps réel
# Le terminal où uvicorn tourne affiche tous les logs

# Augmenter le niveau de log
# Dans .env :
LOG_LEVEL=DEBUG
```

## 📊 Endpoints Disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Health check global |
| `/api/v1/conversation/health` | GET | Health check conversation |
| `/api/v1/conversation/status` | GET | Statut du service |
| `/api/v1/conversation/metrics` | GET | Métriques |
| `/api/v1/conversation/{user_id}` | POST | Endpoint principal |
| `/api/v1/conversation/{user_id}/stream` | POST | Streaming |

## 🔧 Analyse des Bugs

Après avoir lancé `python test_detailed.py`, vous aurez :

1. **Résultats console** : Affichage détaillé de chaque test
2. **Fichier JSON** : Résultats complets pour analyse
3. **Section BUGS** : Liste automatique des bugs trouvés
4. **Section IMPROVEMENTS** : Suggestions d'amélioration

### Exemple d'analyse

```bash
# Lancer les tests
python test_detailed.py

# Regarder les bugs trouvés
cat test_results_*.json | jq '.bugs'

# Regarder les améliorations suggérées
cat test_results_*.json | jq '.improvements'
```

## 🎯 Prochaines Étapes

Une fois que vous avez testé et identifié les bugs :

1. **Partagez les résultats** : Envoyez le fichier `test_results_*.json`
2. **Listez les bugs** : Copiez la section BUGS du terminal
3. **On corrigera ensemble** : Je corrigerai les bugs identifiés
4. **On optimisera** : Améliorations de performance et fiabilité

## 💡 Tips

- **Premier lancement** : Peut prendre 10-30s (initialisation LangChain)
- **Requêtes lentes** : Normal pour les premières requêtes (cold start)
- **Cache** : Les requêtes similaires seront plus rapides
- **JWT** : Pour les tests, vous pouvez utiliser un token dummy

Bon test ! 🚀
