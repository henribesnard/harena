# Harena

## 🚀 Déploiement AWS

### 📚 Documentation Complète de Déploiement

**Pour déployer Harena sur AWS, suivez ces documents dans l'ordre:**

#### 🚀 Pour Déployer MAINTENANT

1. **[COMMANDES_DEPLOIEMENT.md](COMMANDES_DEPLOIEMENT.md)** - ⭐ **DÉPLOIEMENT PRÊT**
   - ✅ Préparation terminée (ports, env, terraform)
   - ✅ Archive créée et uploadée sur S3 (5.8 MB)
   - ✅ Scripts de migration et déploiement prêts
   - 📋 Commandes prêtes à copier-coller
   - ⏱️ Temps estimé: 25-35 minutes
   - **Tout est prêt, vous pouvez déployer immédiatement!**

#### 📖 Documentation de Référence

2. **[RESUME_DEPLOIEMENT.md](RESUME_DEPLOIEMENT.md)** - Vue d'ensemble
   - Architecture cible et services à déployer
   - Plan simplifié sur 3 jours
   - Checklist complète

3. **[ROADMAP_DEPLOIEMENT.md](ROADMAP_DEPLOIEMENT.md)** - Plan détaillé
   - 6 phases de déploiement expliquées
   - Détails techniques de chaque étape
   - Configuration de tous les services systemd

4. **[GUIDE_EXPLOITATION.md](GUIDE_EXPLOITATION.md)** - Guide opérationnel
   - URLs et endpoints de tous les services
   - Démarrer/arrêter l'infrastructure (gestion des coûts)
   - Déployer des mises à jour du code
   - Monitoring, logs, et dépannage
   - Connexion DBeaver à PostgreSQL AWS

5. **[DEPLOIEMENT_AWS.md](DEPLOIEMENT_AWS.md)** - Infrastructure actuelle
   - État de l'infrastructure déployée
   - Services actuellement actifs
   - Configuration technique détaillée

### 🎯 État Actuel

**Infrastructure AWS**: ✅ Créée et opérationnelle
- VPC, EC2, PostgreSQL RDS, Redis ElastiCache, S3 + CloudFront

**Services Backend Déployés**: 2/6
- ✅ Conversation Service (port 8001)
- ✅ Metric Service (port 8004)
- ❌ User Service (port 8000) - À déployer
- ❌ Sync Service (port 8002) - À déployer
- ❌ Enrichment Service (port 8003) - À déployer
- ❌ Search Service (port 8005) - À déployer

**Prochaine étape**: Suivre [ROADMAP_DEPLOIEMENT.md](ROADMAP_DEPLOIEMENT.md) pour déployer les 4 services manquants

## Local Development (Unified Port)

- Single entrypoint: run `python local_app.py` and access `http://localhost:8000`.
- All services (users, sync, enrichment, search, conversation) are mounted under the same FastAPI app.
- Standalone service scripts are disabled by default to avoid port conflicts. To run a service alone, set `HARENA_STANDALONE=true` and invoke the target module, e.g. `HARENA_STANDALONE=true python search_service/main.py`.

### Logging

- Local app forces a deterministic logger configuration to stdout; optional file logging via env.
- Control with:
  - `LOG_LEVEL` (e.g. `INFO`, `DEBUG`)
  - `LOG_TO_FILE` (`True`/`False`)
  - `LOG_FILE` (default `harena_local.log`)

## Organisation du code

Les anciens modules `agents`, `clients`, `models` et `utils` ont été supprimés.
Leur implémentation est désormais centralisée sous le package
`conversation_service`.

Structure simplifiée :

```
conversation_service/
    core/
        conversation_service.py
        transaction_manager.py
    repository.py
    message_repository.py
```

## Configuration

Définissez la variable d'environnement `SECRET_KEY`, utilisée pour la
signature et la vérification des jetons Bearer à travers tous les services.

## Tests unitaires

Avant d'exécuter les tests, installez les dépendances Web de base :

```bash
pip install aiohttp fastapi sqlalchemy pydantic-settings
```

Assurez-vous également que la bibliothèque `autogen-agentchat` est disponible :

```bash
pip install autogen-agentchat
```

Si l'installation n'est pas possible, un stub minimal est fourni dans le dossier
`autogen_agentchat/`. Ajoutez-le au `PYTHONPATH` pour qu'il soit pris en compte :

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

Les tests nécessitent l'extra `ag2[openai]`. Installez les dépendances puis
cet extra :

```bash
pip install -r requirements.txt && pip install "ag2[openai]"
```

Les tests peuvent ensuite être exécutés avec `pytest` :

```bash
pytest
```

### JWT compatibility check

Verify that the token issued by the user service is accepted by the
conversation service with:

```bash
python test_jwt_compatibility.py
```

The script prints `COMPATIBILITÉ TOTALE` when validation succeeds and exits with
a non-zero status on failure.

Pour lancer uniquement les tests du *conversation service* :

```bash
pytest conversation_service/tests
```

Chaque fichier de test peut également être lancé directement :

```bash
python test_metrics_endpoint.py
```

Sans accès réseau, configurez les agents avec `llm_config=False` pour éviter les
appels externes pendant les tests.

## Test du détecteur d'intentions avec un modèle personnalisé

Le script `test_model.py` ne propose plus de menu interactif. Il fonctionne en
mode mock par défaut et se configure via des flags en ligne de commande. Pour
charger un modèle HuggingFace ou local, utilisez `--use-model` et indiquez
éventuellement un nom de modèle :

```bash
# Mode mock uniquement
python test_model.py

# Utiliser un modèle spécifique
python test_model.py --use-model --model-name my-org/mon-modele
# ou
MODEL_NAME=/chemin/vers/mon-modele python test_model.py --use-model

# Activer les traces de débogage
python test_model.py --use-model --debug
```


Pour un benchmark d'intention, consultez [README_intent_benchmark.md](README_intent_benchmark.md).

## Service startup

The conversation service now performs a full agent health check during
startup. If any agent reports an unhealthy status, initialization fails and
the process exits instead of running in a degraded state.
### Usage

Run the conversation service with [Uvicorn](https://www.uvicorn.org/):

```bash
uvicorn conversation_service.main:app --reload
```

The FastAPI app is built by `create_app()` in `conversation_service/main.py`.


## Table `conversation_messages`

La table `conversation_messages` enregistre chaque message individuel échangé dans une conversation.

| Colonne | Type | Exemple |
| --- | --- | --- |
| `id` | `int` | `1` |
| `conversation_id` | `int` (FK `conversations.id`) | `42` |
| `user_id` | `int` (FK `users.id`) | `7` |
| `role` | `varchar(20)` | `'user'` |
| `content` | `text` | `'Bonjour'` |
| `created_at` | `timestamp` | `2025-08-21T22:54:06Z` |
| `updated_at` | `timestamp` | `2025-08-21T22:54:06Z` |

### Exemple d’utilisation

```sql
INSERT INTO conversation_messages (conversation_id, user_id, role, content)
VALUES (42, 7, 'user', 'Bonjour');
```

### Migration

```bash
alembic upgrade 5038c27e9983
```

### Rollback

```bash
alembic downgrade 6eb09f813ccf
```


## Local data seed

For development, populate the `harena_transactions` Elasticsearch index with a
few sample debit and credit transactions for 2025:

```bash
./scripts/dev_init.sh
```

This script runs `search_service/scripts/seed_transactions.py` and requires an
Elasticsearch instance accessible via the `BONSAI_URL` environment variable
or a local node at `http://localhost:9200`.

## Elasticsearch index migration

The `harena_transactions` index now includes account metadata
(`account_name`, `account_type`, `account_balance`,
`account_currency_code`) and custom analyzers (`french_financial`,
`merchant_analyzer`). Each text field also exposes a `.keyword` subfield
for exact filtering.

To migrate existing data into the new mapping run:

```bash
python scripts/migrate_es_index.py
```

The script creates a new `<index>_v2` index with the updated mapping and
reindexes all documents from the current index.

## Search Service models

Les modèles Pydantic exposés par le Search Service se trouvent dans le dossier
[`search_service/models`](search_service/models). Le fichier
[`request.py`](search_service/models/request.py) décrit la forme attendue des
requêtes tandis que [`response.py`](search_service/models/response.py) détaille
la réponse standardisée. Chaque modèle inclut un exemple complet de payload via
`json_schema_extra`.

## Authentication

Most endpoints, including `POST /api/v1/conversation/{user_id}`, require an
OAuth2 Bearer token. Clients must authenticate against `/users/auth/login` and
include the returned `access_token` in an `Authorization: Bearer <token>`
header on their first request. The test clients in this repository obtain a
token during initialization so that every call is properly authenticated.

## Environment Variables

The application loads configuration through Pydantic `Settings` classes. The
following environment variables are recognised:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`: configuration for OpenAI access
- `BONSAI_URL`: Elasticsearch endpoint used by search and enrichment services
- `DATABASE_URL`: connexion PostgreSQL
- `POSTGRES_SERVER`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`: variables alternatives pour définir la base si `DATABASE_URL` n'est pas fourni
- `REDIS_URL`: Redis cache connexion (optionally `REDISCLOUD_URL`)
- `BRIDGE_CLIENT_ID`, `BRIDGE_CLIENT_SECRET`: Bridge API credentials
- `DEEPSEEK_API_KEY`: key for the DeepSeek client
- `CORS_ORIGINS`, `HOST`, `PORT`, `DEBUG`: generic service settings
- `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE`: logging controls for local dev
- `HARENA_STANDALONE`: set to `true` to allow launching individual service entrypoints

`SEARCH_SERVICE_URL` sets the base URL for the Search Service. It defaults to
`http://localhost:8000/api/v1/search` and the service automatically appends
`/search` when issuing queries.

The `USE_MOCK_INTENT_AGENT` toggle has been removed. To exercise the
`MockIntentAgent` (for example during tests), inject it explicitly instead of
setting an environment variable.

## `metadata.workflow_data`

The conversation API (`POST /api/v1/conversation/{user_id}`) includes a
`metadata` object in its response. A nested `workflow_data` dictionary exposes
details collected during the multi‑agent workflow:

| Key | Type | Description |
| --- | --- | --- |
| `user_message` | `str` | Original message provided by the user |
| `conversation_id` | `str` | Identifier of the conversation |
| `intent_result` | `IntentResult` | Structured intent detection output |
| `search_results` | `AgentResponse \| list \| None` | Raw search results returned by the `SearchQueryAgent` (optional) |
| `final_response` | `str \| None` | Final text returned to the user |
| `search_error` | `bool` | Indicates whether the search step failed |
| `search_results_count` | `int` | Number of items found by the search agent |

The key `search_results_count` is always present and can be used to quickly know
how many results were returned without inspecting the full `search_results`
payload.

## Intentions supportées

Le `LLMIntentAgent` répond toujours avec un objet JSON structuré et ne produit
jamais de texte libre. Le tableau suivant récapitule toutes les intentions
disponibles dans Harena ; il est également disponible dans
[INTENTS.md](INTENTS.md) pour référence.

### Transactions

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| TRANSACTION_SEARCH | FINANCIAL_QUERY | Rechercher toutes transactions sans filtre | `["list_transactions"]` |
| SEARCH_BY_DATE | FINANCIAL_QUERY | Transactions pour une date/période | `["search_by_date","list_transactions"]` |
| SEARCH_BY_AMOUNT | FINANCIAL_QUERY | Transactions par montant | `["filter_by_amount","list_transactions"]` |
| SEARCH_BY_MERCHANT | FINANCIAL_QUERY | Transactions liées à un marchand précis | `["search_by_merchant","list_transactions"]` |
| SEARCH_BY_CATEGORY | FINANCIAL_QUERY | Transactions par catégorie | `["search_by_category","list_transactions"]` |
| SEARCH_BY_AMOUNT_AND_DATE | FINANCIAL_QUERY | Combinaison montant + date | `["filter_by_amount","search_by_date","list_transactions"]` |
| SEARCH_BY_OPERATION_TYPE | FINANCIAL_QUERY | Transactions filtrées par type d’opération (débit, crédit, carte…) | `["filter_by_operation_type","list_transactions"]` |
| SEARCH_BY_TEXT | FINANCIAL_QUERY | Recherche textuelle libre | `["search_by_text","list_transactions"]` |
| COUNT_TRANSACTIONS | FINANCIAL_QUERY | Compter les transactions correspondant à une requête | `["count_transactions"]` |
| MERCHANT_INQUIRY | FINANCIAL_QUERY | Analyse détaillée par marchand | `["search_by_merchant","merchant_breakdown"]` |
| FILTER_REQUEST | FILTER_REQUEST | Raffiner une requête transactionnelle (ex. uniquement débits) | `["apply_filters"]` |

### Analyse de dépenses

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| SPENDING_ANALYSIS | SPENDING_ANALYSIS | Analyse globale des dépenses | `["calculate_total","spending_breakdown"]` |
| SPENDING_ANALYSIS_BY_CATEGORY | SPENDING_ANALYSIS | Analyse par catégorie | `["calculate_total","spending_breakdown"]` |
| SPENDING_ANALYSIS_BY_PERIOD | SPENDING_ANALYSIS | Analyse par période | `["calculate_total","time_breakdown"]` |
| SPENDING_COMPARISON | SPENDING_ANALYSIS | Comparaison de périodes ou de catégories | `["compare_periods","compare_categories"]` |
| TREND_ANALYSIS | TREND_ANALYSIS | Tendance/évolution des dépenses | `["trend_analysis","monthly_comparison"]` |
| CATEGORY_ANALYSIS | SPENDING_ANALYSIS | Répartition par catégories | `["category_breakdown","spending_distribution"]` |
| COMPARISON_QUERY | SPENDING_ANALYSIS | Comparaison ciblée (ex. restaurants vs courses) | `["compare_categories","budget_breakdown"]` |

### Soldes de comptes

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| BALANCE_INQUIRY | ACCOUNT_BALANCE | Solde général actuel | `["get_current_balance"]` |
| ACCOUNT_BALANCE_SPECIFIC | ACCOUNT_BALANCE | Solde d’un compte précis | `["get_account_balance"]` |
| BALANCE_EVOLUTION | ACCOUNT_BALANCE | Historique/évolution du solde | `["show_balance_trend"]` |

### Intentions conversationnelles

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| GREETING | GREETING | Bonjour, Salut… | `["greeting_response"]` |
| CONFIRMATION | CONFIRMATION | Merci, parfait… | `["acknowledgment_response"]` |
| CLARIFICATION | CLARIFICATION | Peux-tu préciser ? | `["clarification_request"]` |
| GENERAL_QUESTION | GENERAL_QUESTION | Question générale ne correspondant à aucune autre intention | `["general_response"]` |

### Intentions non supportées

| Intent Type | Category | Example | Actions |
| --- | --- | --- | --- |
| TRANSFER_REQUEST | UNSUPPORTED | Faire un virement | `[]` |
| PAYMENT_REQUEST | UNSUPPORTED | Payer une facture | `[]` |
| CARD_BLOCK | UNSUPPORTED | Bloquer ma carte | `[]` |
| BUDGET_INQUIRY | UNSUPPORTED (future) | Où en est mon budget ? | `[]` |
| GOAL_TRACKING | UNSUPPORTED (future) | Objectifs d’épargne | `[]` |
| EXPORT_REQUEST | UNSUPPORTED (future) | Exporter transactions | `[]` |
| OUT_OF_SCOPE | UNSUPPORTED | Requête hors domaine (ex. recette de cuisine) | `[]` |

### Intentions ambiguës ou erreurs

| Intent Type | Category | Description | Suggested actions |
| --- | --- | --- | --- |
| UNCLEAR_INTENT | UNKNOWN | Intention ambiguë ou non reconnue | `["ask_to_rephrase"]` |
| UNKNOWN | UNKNOWN | Phrase inintelligible | `["ask_to_rephrase"]` |
| TEST_INTENT | UNKNOWN | Message de test («[TEST] ping») | `["no_action"]` |
| ERROR | UNKNOWN | Entrée corrompue | `["retry_or_contact_support"]` |

Entity types must be one of the values from `EntityType` in
`conversation_service/models/financial_models.py`.

Example response:

```json
{
  "intent": "SEARCH_BY_MERCHANT",
  "confidence": 0.92,
  "entities": [{"entity_type": "MERCHANT", "value": "AMAZON"}]
}
```

## Entités de recherche et filtres Elasticsearch

Le `SearchQueryAgent` normalise certaines entités afin de construire des filtres
Elasticsearch. Le tableau suivant décrit la valeur canonique produite pour
chaque entité reconnue et le filtre appliqué :

| Entité (`EntityType`) | Valeur canonique | Filtre ES |
| --- | --- | --- |
| `CATEGORY` | terme anglais normalisé (`virement(s)` → `transfer`) | `category_name` |
| `OPERATION_TYPE` | terme anglais normalisé | `operation_type` |
| `TRANSACTION_TYPE` | termes anglais normalisés (liste unique) | `transaction_types` |
| `DATE_RANGE` | `{ "gte": "...", "lte": "..." }` | `date` |
| `DATE` | convertie en `YYYY-MM` ou `YYYY-MM-DD` puis en plage | `date` |
| `RELATIVE_DATE` | résolue via mots‑clés (`current_month`, `current_week`…) | `date` |
| `AMOUNT` | nombre avec marge de ±10 % ou comparaison absolue | `amount` / `amount_abs` |

> Les entités non listées ne créent pas de filtres et servent uniquement à
> enrichir la recherche textuelle.

## Restes à faire

- Trouver une solution efficace pour la détection d'intentions.
- Revoir le workflow de synchronisation : est-il nécessaire d'enregistrer les données dans une base relationnelle avant de les propager vers Elasticsearch ?
- Réduire la latence des réponses.
- Ajouter un endpoint pour les réponses en streaming.

