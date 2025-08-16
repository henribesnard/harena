# Intent Benchmark

Ce document explique comment exécuter le script `intent_benchmark.py`.

## Dépendances

- `openai`
- `python-dotenv`
- `jsonschema`

## Variable d'environnement

Avant d'exécuter le script, définissez la variable d'environnement `OPENAI_API_KEY` avec votre clé API OpenAI.

## Utilisation

Installez les dépendances puis lancez :

```bash
python intent_benchmark.py --model <nom_du_modèle>
```

Remplacez `<nom_du_modèle>` par le modèle OpenAI ou local souhaité.

