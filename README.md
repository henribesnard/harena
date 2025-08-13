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
