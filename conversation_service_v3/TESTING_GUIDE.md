# Guide de test - conversation_service_v3

## 🔐 Authentification

### Obtenir un token JWT

```bash
curl --request POST \
  --url http://localhost:3000/api/v1/users/auth/login \
  --header 'content-type: multipart/form-data' \
  --form username=henri@example.com \
  --form password=Henri123456
```

**Réponse attendue**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": 123
}
```

**Extraire le token**:
```bash
# Linux/Mac
export TOKEN=$(curl -s --request POST \
  --url http://localhost:3000/api/v1/users/auth/login \
  --header 'content-type: multipart/form-data' \
  --form username=henri@example.com \
  --form password=Henri123456 | jq -r '.access_token')

# Windows PowerShell
$response = Invoke-RestMethod -Uri "http://localhost:3000/api/v1/users/auth/login" -Method POST -Form @{username="henri@example.com"; password="Henri123456"}
$TOKEN = $response.access_token
```

---

## 🧪 Tests des améliorations

### Prérequis

1. **Services démarrés**:
   ```bash
   docker-compose up -d
   ```

2. **Vérifier que conversation_service_v3 est actif**:
   ```bash
   curl http://localhost:3008/health
   ```

3. **Obtenir un token** (voir section ci-dessus)

---

## 📝 Scénarios de test

### Test 1: Filtre montant "plus de X"

**Objectif**: Vérifier que "plus de X euros" utilise `{"gt": X}` (EXCLUT X)

**Question**: "Mes dépenses de plus de 100 euros"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Mes dépenses de plus de 100 euros"
  }'
```

**Points de validation**:
- ✅ Query générée contient `"amount_abs": {"gt": 100}`
- ✅ Agrégations incluent `total_amount` et `transaction_count`
- ✅ Réponse commence par le total des agrégations

---

### Test 2: Question analytique avec catégorie

**Objectif**: Vérifier les agrégations et la réponse basée sur les totaux

**Question**: "Combien j'ai dépensé en restaurants ce mois?"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Combien j'\''ai dépensé en restaurants ce mois?"
  }'
```

**Points de validation**:
- ✅ Filtre catégorie: `"category_name": {"match": "restaurant"}`
- ✅ Filtre date du mois en cours
- ✅ Agrégations: `total_spent`, `transaction_count`, `avg_transaction`
- ✅ Réponse indique le total exact des agrégations (pas seulement les exemples)

---

### Test 3: Évolution temporelle

**Objectif**: Vérifier l'utilisation du template monthly_trend

**Question**: "Évolution de mes dépenses sur les 6 derniers mois"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Évolution de mes dépenses sur les 6 derniers mois"
  }'
```

**Points de validation**:
- ✅ Agrégation `date_histogram` avec `calendar_interval: "month"`
- ✅ Date range couvre 6 mois
- ✅ Réponse formate correctement l'évolution temporelle
- ✅ Formatage avec émojis 📅 et structure claire

---

### Test 4: Répartition par catégorie

**Objectif**: Vérifier l'utilisation du template by_category

**Question**: "Répartition de mes dépenses par catégorie"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Répartition de mes dépenses par catégorie"
  }'
```

**Points de validation**:
- ✅ Agrégation terms sur `category_name.keyword`
- ✅ Size: 20 (toutes les catégories)
- ✅ Sous-agrégations: `total_amount`, `transaction_count`, `avg_transaction`
- ✅ Réponse affiche toutes les catégories (pas seulement les exemples de transactions)
- ✅ Formatage avec "🏷️" et nombre total de groupes

---

### Test 5: Top marchands

**Objectif**: Vérifier le template top_merchants

**Question**: "Où est-ce que je dépense le plus?"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Où est-ce que je dépense le plus?"
  }'
```

**Points de validation**:
- ✅ Agrégation terms sur `merchant_name.keyword`
- ✅ Order par `total_spent` desc
- ✅ Sous-agrégations: `total_spent`, `frequency`, `avg_basket`
- ✅ Réponse liste les top marchands avec montants et fréquence

---

### Test 6: Filtre "au moins X"

**Objectif**: Vérifier que "au moins X" utilise `{"gte": X}` (INCLUT X)

**Question**: "Mes dépenses d'au moins 50 euros"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Mes dépenses d'\''au moins 50 euros"
  }'
```

**Points de validation**:
- ✅ Query contient `"amount_abs": {"gte": 50}` (pas "gt")
- ✅ Les transactions de exactement 50€ sont incluses

---

### Test 7: Multi-critères

**Objectif**: Vérifier la combinaison de plusieurs filtres

**Question**: "Mes achats chez Carrefour de plus de 30 euros ce mois"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Mes achats chez Carrefour de plus de 30 euros ce mois"
  }'
```

**Points de validation**:
- ✅ Filtre merchant: `{"match": "Carrefour"}`
- ✅ Filtre montant: `{"gt": 30}`
- ✅ Filtre date du mois en cours
- ✅ Tous les filtres combinés correctement

---

### Test 8: Statistiques globales

**Objectif**: Vérifier le template spending_statistics

**Question**: "Résumé de mes finances ce mois"

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation \
  --header "Authorization: Bearer $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "user_id": 123,
    "message": "Résumé de mes finances ce mois"
  }'
```

**Points de validation**:
- ✅ Agrégation `stats` pour statistiques complètes
- ✅ Filtres séparés pour debit et credit
- ✅ Réponse affiche: count, sum, avg, min, max
- ✅ Formatage avec "📈" pour les stats

---

## 🔍 Validation de la qualité des réponses

### Critères de qualité

Pour chaque test, vérifier que la réponse:

1. **Utilise les agrégations**:
   - ✅ Les chiffres totaux proviennent des agrégations
   - ✅ Ne se limite pas aux exemples de transactions
   - ✅ Mentionne "X transactions" basé sur les agrégations

2. **Format clair**:
   - ✅ Commence par la réponse directe
   - ✅ Inclut des statistiques clés
   - ✅ Formatage agréable avec émojis modérés
   - ✅ Exemples de transactions si pertinent

3. **Précision**:
   - ✅ Pas de confusion entre totaux et exemples
   - ✅ Utilise "vos/votre" (pas "utilisateur 123")
   - ✅ Montants et comptages corrects

---

## 📊 Logs de débogage

### Activer les logs détaillés

```bash
# Voir les logs de conversation_service_v3
docker logs -f harena_conversation_v3

# Filtrer pour voir les queries générées
docker logs harena_conversation_v3 2>&1 | grep "Query built"

# Voir les agrégations
docker logs harena_conversation_v3 2>&1 | grep "RÉSUMÉ STATISTIQUE"
```

### Points de contrôle dans les logs

1. **Query building**:
   ```
   Query built successfully with function calling
   ```

2. **Agrégations formatées**:
   ```
   📊 RÉSUMÉ STATISTIQUE COMPLET (SOURCE DE VÉRITÉ)
   ```

3. **Réponse générée**:
   ```
   Response generated successfully. Length: XXX chars
   ```

---

## 🐛 Débogage des problèmes courants

### Problème: "plus de X" génère "gte" au lieu de "gt"

**Cause**: Prompt système mal interprété

**Solution**: Vérifier le prompt dans `elasticsearch_builder_agent.py`:
```python
1. "plus de X euros" = amount_abs: {{"gt": X}} → EXCLUT X (strictement supérieur)
2. "au moins X euros" = amount_abs: {{"gte": X}} → INCLUT X (supérieur ou égal)
```

### Problème: Réponse basée sur les exemples au lieu des agrégations

**Cause**: ResponseGenerator ne priorise pas les agrégations

**Solution**: Vérifier le prompt dans `response_generator_agent.py`:
```
IMPORTANT - Utilisation des données:
- Les AGRÉGATIONS contiennent les STATISTIQUES GLOBALES sur TOUS les résultats
- TOUJOURS utiliser les AGRÉGATIONS pour les chiffres totaux
```

### Problème: Agrégations manquantes dans la réponse

**Cause**: Template d'agrégation non utilisé

**Solution**: Vérifier que le prompt inclut les templates:
```python
templates_description = get_all_templates_description()
```

---

## ✅ Checklist de validation

Après avoir exécuté tous les tests:

- [ ] Test 1: Filtre "plus de" → `{"gt": X}`
- [ ] Test 2: Question analytique → Agrégations correctes
- [ ] Test 3: Évolution temporelle → date_histogram
- [ ] Test 4: Répartition catégorie → Template by_category
- [ ] Test 5: Top marchands → Template top_merchants
- [ ] Test 6: Filtre "au moins" → `{"gte": X}`
- [ ] Test 7: Multi-critères → Tous les filtres combinés
- [ ] Test 8: Statistiques globales → Template spending_statistics

- [ ] Les réponses utilisent les agrégations (pas seulement les exemples)
- [ ] Le formatage des agrégations est enrichi (emojis, contexte)
- [ ] Les prompts incluent les templates et exemples
- [ ] Pas d'erreurs dans les logs

---

## 📚 Ressources

- `function_definitions.py` - Définitions des fonctions et templates
- `IMPROVEMENTS_APPLIED.md` - Documentation des améliorations
- `function_calling_examples.md` - 50+ exemples de questions
- `v3_implementation_plan.md` - Plan d'implémentation complet

---

## 💡 Questions de test supplémentaires

Pour tester davantage les améliorations:

```bash
# Test 9: Pattern hebdomadaire
"Quel jour de la semaine je dépense le plus?"

# Test 10: Tendance hebdomadaire
"Mes dépenses par semaine ce mois"

# Test 11: Comparaison implicite
"Mes dépenses en courses vs restaurants"

# Test 12: Recherche simple
"Mes 10 dernières transactions"

# Test 13: Marchand spécifique
"Combien j'ai dépensé chez Amazon?"

# Test 14: Plage de montants
"Mes dépenses entre 20 et 50 euros"

# Test 15: Type de transaction
"Mes revenus ce mois"
```

Chaque question doit générer:
- Une query au format search_service correct
- Des agrégations appropriées
- Une réponse basée sur les agrégations (pas les exemples)
