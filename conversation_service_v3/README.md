# Conversation Service V3 🚀

**Architecture LangChain avec Agents Autonomes + Streaming SSE + Redis Memory**

Architecture révolutionnaire basée sur des agents LangChain autonomes avec capacité d'auto-correction, streaming temps réel, persistence des conversations et optimisations avancées du contexte LLM.

## 🎯 Fonctionnalités V3

### 🔄 Streaming Server-Sent Events (SSE)
- **Réponses en temps réel**: Streaming progressif des réponses via SSE
- **Progressive status messages**: Messages de progression contextuels pendant le traitement
  - "• Analyse de votre question..."
  - "• Recherche de vos transactions..."
  - "• Analyse de vos données..."
  - "• X transaction(s) trouvée(s), génération de la réponse..."
- **Auto-hide**: Les messages de statut disparaissent automatiquement dès le début du streaming de la réponse
- **Zero latency impact**: Messages émis pendant le traitement réel, sans ajout de délai

### 💾 Redis Conversation Memory
- **Persistence des conversations**: Sauvegarde automatique dans Redis avec TTL configurable (24h par défaut)
- **Token control**: Limite intelligente du contexte (4000 tokens par défaut) pour optimiser les coûts LLM
- **Truncation automatique**: Suppression des tours les plus anciens si dépassement de la limite
- **Context enrichment**: Historique conversationnel injecté dans le prompt LLM pour cohérence

### 🎯 Classification d'Intent Sémantique
- **Intentions supportées**:
  - `search`: Recherche de transactions spécifiques
  - `aggregate`: Calculs et agrégations (totaux, moyennes, statistiques)
  - `analytical`: Analyses avancées avec 9 types d'analyses
- **9 Types d'analyses analytiques**:
  - `temporal_trends`: Tendances temporelles (évolution dans le temps)
  - `category_breakdown`: Répartition par catégories
  - `spending_patterns`: Patterns de dépenses (récurrence, saisonnalité)
  - `comparison`: Comparaisons (périodes, catégories, marchands)
  - `anomaly_detection`: Détection d'anomalies et valeurs atypiques
  - `budget_analysis`: Analyse budgétaire et dépassements
  - `merchant_analysis`: Analyse par marchand (fréquence, montants)
  - `prediction`: Prédictions et projections futures
  - `custom`: Analyses personnalisées
- **Confidence scoring**: Chaque classification avec score de confiance

### 🔍 Filtrage Sémantique Strict
- **Filtrage "achats"**: Distinction stricte entre achats (débits) et autres transactions
- **Exclusion automatique**: Les virements, salaires, remboursements sont exclus des "achats"
- **Transaction type control**: Gestion précise des types `debit` vs `credit`

### 🎨 Optimisation du Contexte LLM
- **Field filtering**: Réduction à 8 champs essentiels des transactions
  - `transaction_id`, `date`, `merchant_name`, `amount`, `category_name`, `transaction_type`, `primary_description`, `user_id`
- **Smart aggregations**: Agrégations priorisées dans le contexte
- **Transaction ordering**: Garantie de l'ordre chronologique des transactions dans les conversations
- **Context window management**: Gestion intelligente de la fenêtre de contexte

### 🛡️ Robustesse et Fiabilité
- **None value handling**: Gestion correcte des valeurs `None` explicites dans Elasticsearch
- **Auto-correction**: Les agents corrigent automatiquement les erreurs de query
- **Validation stricte**: Validation du `user_id` dans toutes les queries pour sécurité
- **Error recovery**: Récupération automatique sur erreurs temporaires

### 🎭 Agents Autonomes
- **QueryAnalyzerAgent**: Analyse la requête et extrait les entités avec classification d'intent
- **ElasticsearchBuilderAgent**: Traduit en query Elasticsearch valide avec auto-correction
- **ResponseGeneratorAgent**: Génère des réponses naturelles avec streaming et insights

## 🚀 Pipeline de Traitement

```
User Query
    ↓
[1. Intent Classification]
    - search / aggregate / analytical
    - Confidence scoring
    ↓
[2. Query Analysis]
    - Entity extraction
    - Semantic filtering
    ↓
[3. Elasticsearch Query Building]
    - Query generation
    - Field filtering
    - Auto-correction if needed
    ↓
[4. Search Execution]
    - Execute on search_service
    - Transaction ordering
    ↓
[5. Response Generation (Streaming)]
    - Context enrichment with Redis history
    - Progressive status messages
    - Real-time SSE streaming
    ↓
[6. Conversation Persistence]
    - Save to Redis
    - Token control & truncation
    ↓
Natural Language Response (Streamed)
```

## 📡 API Endpoints

### POST /api/v3/conversation/stream (Streaming SSE)
Endpoint principal avec streaming temps réel.

**Request:**
```json
{
  "message": "Analyse mes revenus de plus de 2500 euros",
  "conversation_id": "123"  // Optionnel, auto-créé si absent
}
```

**Headers:**
```
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json
```

**Response (SSE Stream):**
```
data: {"type": "status", "message": "• Analyse de votre question..."}

data: {"type": "status", "message": "• Recherche de vos transactions..."}

data: {"type": "status", "message": "• Analyse de vos données..."}

data: {"type": "status", "message": "• 21 transaction(s) trouvée(s), génération de la réponse..."}

data: {"type": "chunk", "content": "Vous avez un total de "}

data: {"type": "chunk", "content": "**21 transactions** "}

data: {"type": "chunk", "content": "avec des revenus dépassant 2500 €..."}

data: {"type": "conversation_id", "conversation_id": 123}

data: {"type": "done"}
```

### GET /api/v3/conversation/conversations/{conversation_id}
Récupère l'historique d'une conversation avec ses tours.

**Response:**
```json
{
  "id": 123,
  "user_id": 3,
  "created_at": "2025-10-23T14:30:00",
  "updated_at": "2025-10-23T14:35:00",
  "turns": [
    {
      "id": 1,
      "user_message": "Mes revenus de plus de 2500 euros",
      "assistant_response": "Vous avez un total de...",
      "created_at": "2025-10-23T14:30:00"
    }
  ]
}
```

### POST /api/v1/conversation/{user_id} (Compatibilité V1)
Endpoint de compatibilité avec conversation_service v1 (non-streaming).

### GET /api/v3/conversation/conversations
Liste toutes les conversations d'un utilisateur.

**Query params:**
- `limit`: Nombre de conversations (défaut: 50)
- `offset`: Offset pour pagination (défaut: 0)

### DELETE /api/v3/conversation/conversations/{conversation_id}
Supprime une conversation et son historique.

### GET /api/v3/conversation/health
Health check de tous les composants (LLM, search_service, Redis, PostgreSQL).

### GET /api/v3/conversation/stats
Statistiques des agents et du service.

## 🔧 Configuration

### Variables d'Environnement

```bash
# Service
SERVICE_NAME=conversation_service_v3
PORT=3008

# External Services
SEARCH_SERVICE_URL=http://harena_search_service:3001

# Database
DATABASE_URL=postgresql://user:password@db:5432/harena

# Redis
REDIS_URL=redis://harena_redis:6379/0
REDIS_TTL_HOURS=24
REDIS_MAX_TOKENS=4000

# LLM Configuration
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Agent Configuration
MAX_CORRECTION_ATTEMPTS=2
QUERY_TIMEOUT_SECONDS=30

# Field Filtering
FIELD_FILTERING_ENABLED=true
TRANSACTION_FIELDS_TO_KEEP=transaction_id,date,merchant_name,amount,category_name,transaction_type,primary_description,user_id

# Logging
LOG_LEVEL=INFO
```

## 🎨 Exemples d'Utilisation

### Exemple 1: Streaming avec status messages
```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:3008/api/v3/conversation/stream",
    json={"message": "Mes dépenses de plus de 100 euros ce mois-ci"},
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    timeout=30.0
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])

            if data["type"] == "status":
                print(f"Status: {data['message']}")
            elif data["type"] == "chunk":
                print(data["content"], end="", flush=True)
            elif data["type"] == "done":
                print("\n✅ Terminé")
```

### Exemple 2: Conversation multi-tours
```python
# Premier message
response1 = httpx.post(
    "http://localhost:3008/api/v3/conversation/stream",
    json={"message": "Mes dépenses en restaurants ce mois-ci"},
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
conversation_id = extract_conversation_id(response1)

# Deuxième message avec contexte
response2 = httpx.post(
    "http://localhost:3008/api/v3/conversation/stream",
    json={
        "message": "Et le mois dernier ?",
        "conversation_id": conversation_id
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

### Exemple 3: Analyse temporelle
```python
response = httpx.post(
    "http://localhost:3008/api/v3/conversation/stream",
    json={"message": "Montre-moi l'évolution de mes dépenses sur les 6 derniers mois"},
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
# Intent détecté: analytical (temporal_trends)
```

## 📈 Performance

### Métriques Typiques (v5.0.12)
- **Temps de réponse moyen**: 1.5-3 secondes (premier chunk)
- **Streaming latency**: <100ms par chunk
- **Taux de succès**: 98%+
- **Taux de correction**: ~10% des queries
- **Redis hit rate**: ~75% pour conversations multi-tours
- **Token savings**: ~60% grâce au field filtering

### Optimisations Appliquées
- ✅ Field filtering: 8 champs essentiels au lieu de 20+
- ✅ Redis caching: Conversation history en mémoire
- ✅ Token control: Truncation intelligente à 4000 tokens
- ✅ Streaming SSE: First byte en <500ms
- ✅ Status messages: Zero latency (émis pendant processing)
- ✅ None handling: Valeurs par défaut avec `or` operator

## 🔐 Sécurité

- **JWT Authentication**: Token Bearer obligatoire
- **User isolation**: Filtre `user_id` automatique et validé
- **Conversation ownership**: Vérification du propriétaire avant accès
- **Field validation**: Schema Elasticsearch strictement respecté
- **SQL injection protection**: Utilisation de SQLAlchemy ORM
- **Redis namespace**: Isolation par conversation_id

## 🐛 Debugging

### Logs Structurés
```bash
# Production
LOG_LEVEL=INFO

# Development
LOG_LEVEL=DEBUG
```

Logs typiques:
```
2025-10-23 14:30:15 - app.agents.query_analyzer_agent - INFO - Analyzing query: Mes revenus de plus de 2500 euros
2025-10-23 14:30:18 - app.agents.query_analyzer_agent - INFO - Query analysis completed: intent=search, confidence=0.95
2025-10-23 14:30:20 - app.agents.response_generator_agent - INFO - [STREAM] Transactions: 21 → 21 (filtered to 8 fields)
```

### Health Check
```bash
curl http://localhost:3008/api/v3/conversation/health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "llm": "ok",
    "search_service": "ok",
    "redis": "ok",
    "database": "ok"
  }
}
```

## 📊 Versions History

### v6.0.0Stable (2025-10-23)
- ✅ Production-ready avec toutes les fonctionnalités stabilisées
- ✅ Streaming SSE avec status messages
- ✅ Redis conversation memory avec token control
- ✅ Field filtering pour optimisation LLM
- ✅ Robust None handling dans transaction data
- ✅ 9 types d'analyses analytiques
- ✅ Filtrage sémantique strict pour "achats"

### v5.0.12 (2025-10-23)
- 🐛 Fix: Handle None values in transaction fields (merchant_name, category_name)

### v5.0.11 (2025-10-23)
- 🐛 Fix: Add None check for merchant field in transaction formatting

### v5.0.10 (2025-10-23)
- 🎨 Style: Replace emojis with bullet points in status messages

### v5.0.9 (2025-10-23)
- ✨ Feature: Add progressive status messages during streaming

### v5.0.8 (2025-10-23)
- ✨ Feature: Apply field filtering in streaming mode for consistency

### v5.0.7 (2025-10-23)
- ✨ Feature: Advanced analytical capabilities with 9 analysis types

### v5.0.6 (2025-10-23)
- ✨ Feature: Optimize LLM context with transaction field filtering

### v5.0.5 (2025-10-23)
- ✨ Feature: Redis conversation memory with token control

### v5.0.4 (2025-10-23)
- ✨ Feature: Implement strict "achats" semantic filtering

## 🔮 Roadmap v6.1

- [ ] Multi-language support (EN, ES, DE)
- [ ] Advanced analytics dashboard
- [ ] Webhook notifications for long-running queries
- [ ] GraphQL API alternative
- [ ] Prometheus metrics export
- [ ] A/B testing framework for agent prompts
- [ ] Voice input support
- [ ] Export to PDF/Excel

## 📝 License

MIT

## 👥 Auteurs

Équipe Harena - 2025
