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

Le script `test_model.py` utilise par défaut le modèle
`microsoft/Phi-3.5-mini-instruct`. Pour essayer d'autres poids (locaux ou sur
HuggingFace), passez le nom du modèle en argument ou via la variable
d'environnement `MODEL_NAME` :

```bash
python test_model.py --model-name my-org/mon-modele
# ou
MODEL_NAME=/chemin/vers/mon-modele python test_model.py
```


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

`SEARCH_SERVICE_URL` sets the base URL for the Search Service. It defaults to
`http://localhost:8000/api/v1/search` and the service automatically appends
`/search` when issuing queries.

## LLMIntentAgent output

The `LLMIntentAgent` always responds with a strict JSON object and never adds
free-form text. The allowed intent values are:

`SEARCH_BY_TEXT`, `SEARCH_BY_MERCHANT`, `SEARCH_BY_CATEGORY`,
`SEARCH_BY_AMOUNT`, `SEARCH_BY_DATE`, `SEARCH_BY_OPERATION_TYPE`,
`ANALYZE_SPENDING`, `ANALYZE_TRENDS`, `COUNT_TRANSACTIONS`,
`TRANSACTION_SEARCH`, `SPENDING_ANALYSIS`, `BALANCE_INQUIRY`,
`GENERAL_QUESTION`, `GREETING`, `OUT_OF_SCOPE`.

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

