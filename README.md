# Harena

## Tests unitaires

Les tests peuvent être exécutés avec `pytest` :

```bash
pytest
```

Chaque fichier de test peut également être lancé directement :

```bash
python test_metrics_endpoint.py
```

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

## Authentication

Most endpoints, including `/conversation/chat`, require an OAuth2 Bearer token.
Clients must authenticate against `/users/auth/login` and include the returned
`access_token` in an `Authorization: Bearer <token>` header on their first
request. The test clients in this repository obtain a token during
initialization so that every call is properly authenticated.

## Environment Variables

`OPENAI_API_KEY` provides the API key used by the `LLMIntentAgent` for intent
detection. If not set, the agent falls back to the DeepSeek key.

`SEARCH_SERVICE_URL` sets the base URL for the Search Service. It defaults to
`http://localhost:8000/api/v1/search` and the service automatically appends
`/search` when issuing queries.

## `metadata.workflow_data`

The conversation API (`/conversation/chat`) includes a `metadata` object in its
response. A nested `workflow_data` dictionary exposes details collected during
the multi‑agent workflow:

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

## Restes à faire

- Trouver une solution efficace pour la détection d'intentions.
- Revoir le workflow de synchronisation : est-il nécessaire d'enregistrer les données dans une base relationnelle avant de les propager vers Elasticsearch ?
- Réduire la latence des réponses.
- Ajouter un endpoint pour les réponses en streaming.

