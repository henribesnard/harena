# Corrections pour v4.1.0

**Tag de base:** v4.0.0
**Date:** 2025-10-16
**Objectif:** Corriger les régressions et améliorer la précision des requêtes LLM

## Problèmes Identifiés

### 1. B016/B017: Liste des marchands manquante dans la query

**Symptôme:** La liste des marchands n'apparaît pas dans la requête générée
**Fichiers concernés:**
- `conversation_service/core/template_engine.py` - ligne 241-252 (gestion merchant_name)
- Templates de requête dans `conversation_service/templates/query/transaction_search/`

**Analyse:**
- Le template_engine gère déjà les agrégations de marchands (ligne 1060+)
- Le problème est probablement dans l'extraction des entités ou dans le mapping des templates
- Vérifier que `merchant_name` est bien passé comme liste dans les filtres de la query

**Solution proposée:**
1. Vérifier les logs de `_prepare_template_context` pour voir si `merchant_name` est extrait
2. S'assurer que le template utilise bien `filters.merchant_name` et non `filters.merchant`
3. Vérifier que la liste des marchands est bien extraite par l'intent classifier

---

### 2. E005/F017: operation_type "card" mal classifié comme retrait

**Symptôme:** Tous les `operation_type: "card"` ne sont pas des retraits
**Contexte:** Il existe une catégorie "retrait espèces" distincte

**Analyse:**
- `operation_type` peut être: card, transfer, check, debit, etc.
- La catégorie "retrait espèces" est déjà dans la base de données
- Le LLM doit analyser **3 dimensions** pour répondre correctement:
  1. **category_name** (ex: "retrait espèces", "Alimentation", etc.)
  2. **operation_type** (ex: "card", "transfer", "check")
  3. **transaction_type** (ex: "debit", "credit")

**Solution proposée:**
1. Améliorer les prompts système pour que le LLM comprenne ces 3 dimensions
2. Ajouter un champ `query` (texte libre) dans les entités pour les requêtes ambiguës
3. Créer une logique de matching combinée:
   - Si requête = "retraits" → vérifier si c'est:
     - category_name="retrait espèces" (prioritaire)
     - OU operation_type="card" + transaction_type="debit"
     - OU query contient "retrait"

**Fichiers à modifier:**
- `conversation_service/agents/llm/intent_classifier.py` - Prompts système
- `conversation_service/models/conversation/entities.py` - Ajouter champ `query`
- Templates de requête - Supporter le champ `query` pour multi_match Elasticsearch

---

### 3. E015: Vérifier que operation_type "unknown" existe

**Symptôme:** Le LLM génère `operation_type: "unknown"`
**À vérifier:**
- Est-ce que "unknown" existe dans la base de données?
- Quels sont les valeurs valides de operation_type?

**Action immédiate:**
```sql
SELECT DISTINCT operation_type FROM raw_transactions ORDER BY operation_type;
```

**Si "unknown" n'existe pas:**
- Mapper "unknown" → NULL ou valeur par défaut
- Documenter les valeurs valides dans les prompts LLM

---

### 4. F011: category_name trop large pour "achats en ligne"

**Symptôme:**
Requête: "Mes achats en ligne du week-end dernier"
Résultat généré:
```json
"category_name": [
  "Carburant",
  "Transport",
  "Loisirs",
  "Entretien maison",
  "achats en ligne",
  "Alimentation",
  "Vêtements"
]
```

**Analyse:**
- Le LLM génère trop de catégories au lieu de se concentrer sur "achats en ligne"
- Problème de précision dans la génération des entités

**Solution proposée:**
1. Améliorer le prompt système pour être plus précis:
   - "Si l'utilisateur mentionne UNE catégorie spécifique, n'inclure QUE cette catégorie"
   - "N'ajouter plusieurs catégories que si l'utilisateur demande explicitement 'toutes mes dépenses' ou similaire"

2. Ajouter une validation post-génération:
   - Si la requête mentionne explicitement une catégorie → filtrer pour garder SEULEMENT celle-ci
   - Utiliser fuzzy matching entre la requête et les catégories extraites

**Fichiers à modifier:**
- `conversation_service/agents/llm/intent_classifier.py` - Améliorer prompts
- Possiblement ajouter une étape de validation des entités

---

## Plan de Tests

### Tests de Non-Régression
Avant d'appliquer les corrections, tester les cas qui **fonctionnent actuellement**:
- [ ] Questions par catégorie simple
- [ ] Questions par période
- [ ] Questions par montant
- [ ] Questions par type de transaction (debit/credit)

### Tests des Corrections
Après corrections:
- [ ] B016/B017: Liste des marchands apparaît dans la query
- [ ] E005/F017: Retraits classifiés correctement (carte vs espèces)
- [ ] E015: operation_type "unknown" géré correctement
- [ ] F011: Catégorie "achats en ligne" précise (pas de liste large)

---

## Prochaines Étapes

1. ✅ Tag v4.0.0 créé avec état actuel
2. ⏳ Analyser chaque problème en détail
3. ⏳ Implémenter les corrections une par une
4. ⏳ Tester chaque correction isolément
5. ⏳ Tests de non-régression complets
6. ⏳ Tag v4.1.0 avec toutes les corrections

---

## Notes Techniques

### Valeurs operation_type vérifiées (Elasticsearch)

**Confirmé via Elasticsearch:**
- `operation_type="card"`: **4,275 transactions**
  - Exemples: Achats OpenAI, Midjourney, frais bancaires
  - ❌ PAS uniquement des retraits espèces!

**Implication:**
- Une requête "mes retraits" ne doit PAS filtrer sur `operation_type="card"`
- Utiliser plutôt `category_name="retrait espèces"` (si existe) OU `query="retrait"` (texte libre)

### Catégories avec "retrait"
```sql
-- Vérifier les catégories de retrait
SELECT category_name, COUNT(*)
FROM categories c
JOIN raw_transactions rt ON c.category_id = rt.category_id
WHERE c.category_name ILIKE '%retrait%'
GROUP BY category_name;
```

### Architecture LLM actuelle
- **Intent Classifier:** Détermine l'intention (transaction_search, analysis_insights, etc.)
- **Entity Extractor:** Extrait les entités (période, montant, catégorie, merchant, etc.)
- **Template Engine:** Génère la requête Elasticsearch à partir des entités
- **Query Orchestrator:** Coordonne le tout

**Point clé:** Les 3 dimensions (category, operation_type, transaction_type) doivent être bien comprises par l'Intent Classifier
