# Conversation Service V2 - Architecture Simplifiée

Assistant financier intelligent avec Text-to-SQL via DeepSeek API.

## Architecture

Cette version V2 implémente une architecture simplifiée et optimisée basée sur:
- **Agent unique** (pas de multi-agents complexe)
- **DeepSeek API** pour le Text-to-SQL et la génération de réponses
- **PostgreSQL** pour les requêtes structurées
- **Redis** pour le cache (hit rate 60-70%)
- **Row-Level Security** pour la sécurité des données

## Pipeline de Traitement

```
1. Authentification JWT
2. Analyse d'intention (DeepSeek)
3. Génération SQL (DeepSeek)
4. Validation SQL (sécurité)
5. Exécution SQL + Cache Redis
6. Construction du contexte
7. Génération réponse naturelle (DeepSeek)
```

## Structure du Projet

```
conversation_service_v2/
├── app/
│   ├── api/
│   │   └── v2/
│   │       └── endpoints/
│   │           └── conversation.py    # Endpoints /api/v2/*
│   ├── auth/
│   │   └── middleware.py              # Authentification JWT
│   ├── core/
│   │   ├── intent_analyzer.py         # Module 1: Analyse intention
│   │   ├── sql_generator.py           # Module 2: Génération SQL
│   │   ├── sql_validator.py           # Module 3: Validation SQL
│   │   ├── sql_executor.py            # Module 4: Exécution + cache
│   │   ├── context_builder.py         # Module 5: Construction contexte
│   │   └── response_generator.py      # Module 6: Réponse naturelle
│   ├── models/
│   │   ├── requests/
│   │   │   └── conversation_requests.py
│   │   └── responses/
│   │       └── conversation_responses.py
│   ├── services/
│   │   └── conversation_service.py    # Orchestrateur principal
│   ├── config/
│   │   └── settings.py
│   └── main.py                        # Application FastAPI
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

1. Créer un environnement virtuel:
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement:
```bash
cp .env.example .env
# Éditer .env avec vos clés API et configurations
```

4. Lancer l'application:
```bash
python -m app.main
```

L'application sera disponible sur `http://localhost:3003`

## Endpoints API

### POST /api/v2/conversation/{user_id}

Poser une question en langage naturel.

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: application/json
```

**Body:**
```json
{
  "query": "Combien j'ai dépensé en restaurants ce mois-ci ?",
  "context": {
    "conversation_id": "uuid-optional",
    "preferences": {
      "language": "fr",
      "include_visualization": true
    }
  }
}
```

**Response:**
```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": 12345,
  "timestamp": "2025-10-19T14:32:15.123Z",
  "answer": "Ce mois-ci, vous avez dépensé 847,32 € en restaurants...",
  "insights": [
    "Augmentation de 23% par rapport au mois dernier",
    "Budget dépassé de 97€"
  ],
  "recommendations": [
    "Réduire les fast-foods pourrait économiser ~60€/mois"
  ],
  "visualization": {
    "type": "bar_chart",
    "title": "Dépenses restaurants",
    "data": {...}
  },
  "metadata": {
    "execution_time_ms": 987,
    "tokens_used": {"input": 2150, "output": 285, "total": 2435},
    "cost_usd": 0.0012,
    "sql_query": "WITH summary AS...",
    "total_transactions_found": 127,
    "cached": false,
    "model_used": "deepseek-chat"
  }
}
```

### GET /api/v2/health

Health check de l'application.

### GET /api/v2/docs

Documentation interactive Swagger UI.

## Configuration

Toutes les configurations se font via variables d'environnement (fichier `.env`):

- `JWT_SECRET_KEY`: Clé secrète pour les JWT
- `DEEPSEEK_API_KEY`: Clé API DeepSeek
- `POSTGRES_*`: Configuration PostgreSQL
- `REDIS_*`: Configuration Redis

## Sécurité

- **Authentification JWT** obligatoire
- **Row-Level Security (RLS)** PostgreSQL
- **Validation SQL** multi-couches
- **Cloisonnement des données** par user_id
- **Aucune requête destructive** (DELETE, DROP, UPDATE interdits)

## Performance

- **Temps de réponse moyen**: ~1000ms
- **Avec cache Redis**: 300-600ms (60-70% des requêtes)
- **Coût par requête**: ~$0.0012 (DeepSeek)

## Développement

Pour lancer en mode développement avec reload:
```bash
uvicorn app.main:app --reload --port 3003
```

## Documentation Complète

Voir `harena_architecture.md` pour la documentation complète de l'architecture.
