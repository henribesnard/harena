# Prompt Examples

Chaque agent utilise des exemples d'entrée/sortie pour illustrer le format attendu.
Ces few-shots sont stockés dans des fichiers YAML et chargés dynamiquement lors de la
construction du prompt.

Fichiers disponibles :
- `llm_intent_agent_examples.yaml`
- `search_query_agent_examples.yaml`
- `response_agent_examples.yaml`
- `orchestrator_agent_examples.yaml`

Modifier ces fichiers nécessite de mettre à jour les tests associés car le format du prompt est
vérifié automatiquement.
