# Guide d'utilisation des prompts

Cette section centralise les templates de prompts utilisés par le `conversation_service`.

## Structure du dossier

```
conversation_service/
  prompts/
    INTENTION/
      template.jinja
    README.md
```

### Conventions

- Chaque intention possède un dossier dédié dans `prompts`.
- Les noms d'intention reprennent l'identifiant défini dans [INTENTS.md](../../INTENTS.md) en MAJUSCULES.
- Les templates sont écrits en Jinja2 et les variables utilisent le `snake_case`.
- Documenter les paramètres attendus et fournir des exemples de rendu.

## Ajouter une nouvelle intention

1. Définir l'intention et les actions associées dans [INTENTS.md](../../INTENTS.md).
2. Créer le dossier `prompts/<NOUVELLE_INTENTION>/`.
3. Ajouter le template principal `template.jinja` et éventuellement un `README` local.
4. Couvrir le prompt par des tests unitaires.
5. Soumettre la modification via une pull request.

## Mettre à jour un template existant

1. Valider le changement avec l'équipe produit.
2. Modifier le template correspondant et mettre à jour sa documentation.
3. Ajuster ou ajouter des tests.
4. Vérifier les impacts sur les conversations existantes.

## Exemples d'intégration

### Pour les développeurs

```python
from conversation_service.prompts import load_prompt
prompt = load_prompt("SEARCH_BY_DATE", date_range="2024-01")
response = openai_client.chat(prompt)
```

### Pour les Product Managers

1. Proposer une nouvelle intention dans [INTENTS.md](../../INTENTS.md).
2. Décrire les cas d'usage et exemples de dialogue dans le dossier `prompts/<INTENTION>/`.
3. Collaborer avec les développeurs pour implémenter et tester le prompt.
