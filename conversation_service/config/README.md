# Configuration du service de conversation

Ce module centralise la configuration liée aux agents et à l'utilisation
de l'API OpenAI.

## Variables d'environnement

| Variable            | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `OPENAI_API_KEY`              | Clé d'API OpenAI utilisée par les agents.                   |
| `OPENAI_BASE_URL`             | URL de base de l'API (défaut `https://api.openai.com/v1`).  |
| `SEARCH_SERVICE_URL`          | URL de l'API du service de recherche.                        |
| `REDIS_URL`                   | URL de connexion Redis.                                     |
| `REDIS_PASSWORD`              | Mot de passe Redis optionnel.                               |
| `REDIS_DB`                    | Index de base de données Redis à utiliser.                  |
| `REDIS_MAX_CONNECTIONS`       | Nombre maximal de connexions Redis.                         |
| `REDIS_HEALTH_CHECK_INTERVAL` | Intervalle de vérification de santé Redis (s).              |
| `REDIS_RETRY_ON_TIMEOUT`      | Réessayer en cas de timeout Redis (`true`/`false`).         |
| `REDIS_CACHE_PREFIX`          | Préfixe des clés cache Redis.                               |
| `LOG_LEVEL`                   | Niveau de log (`INFO`, `DEBUG`, ...).                       |
| `LOG_FORMAT`                  | Format de log compatible `logging`.                         |

La fonction `reload_settings()` permet de recharger ces variables à chaud
sans redémarrer le service.

## Configuration des agents

Le fichier [`autogen_config.py`](autogen_config.py) définit deux agents
par défaut :

- **assistant** : modèle `gpt-4o-mini`.
- **reasoner** : modèle `gpt-4o` avec température plus basse.

Les détails des modèles (coûts, limites de rate‑limiting, modèle de
secours) sont documentés dans [`openai_config.py`](openai_config.py).
