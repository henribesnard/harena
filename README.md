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
