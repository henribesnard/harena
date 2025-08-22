# Configuration du service de conversation

Ce module centralise la configuration liée aux agents et à l'utilisation
de l'API OpenAI.

## Variables d'environnement

| Variable            | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `OPENAI_API_KEY`    | Clé d'API OpenAI utilisée par les agents.                   |
| `OPENAI_BASE_URL`   | URL de base de l'API (défaut `https://api.openai.com/v1`).  |
| `REDIS_URL`         | URL de connexion Redis.                                     |
| `REDIS_PASSWORD`    | Mot de passe Redis optionnel.                               |
| `LOG_LEVEL`         | Niveau de log (`INFO`, `DEBUG`, ...).                       |
| `LOG_FORMAT`        | Format de log compatible `logging`.                         |

La fonction `reload_settings()` permet de recharger ces variables à chaud
sans redémarrer le service.

## Configuration des agents

Le fichier [`autogen_config.py`](autogen_config.py) définit deux agents
par défaut :

- **assistant** : modèle `gpt-4o-mini`.
- **reasoner** : modèle `gpt-4o` avec température plus basse.

Les détails des modèles (coûts, limites de rate‑limiting, modèle de
secours) sont documentés dans [`openai_config.py`](openai_config.py).
