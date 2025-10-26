# 🧪 Instructions pour tester DeepSeek

## Résultats du test OpenAI

✅ **OpenAI a été testé avec succès !**

- **Latence:** 42.92 secondes
- **Qualité:** Excellente (réponse structurée, personnalisée, pertinente)
- **Longueur:** 2,908 caractères
- **Résultats:** Voir `openai_test_result.json` et `RAPPORT_LLM_COMPARISON.md`

---

## Pour tester DeepSeek et comparer

### Étape 1 : Obtenir une clé API DeepSeek

1. Aller sur https://platform.deepseek.com
2. Créer un compte (ou se connecter)
3. Générer une clé API
4. Copier la clé (format: `sk-xxxxxxxxxxxxx`)

### Étape 2 : Configurer la clé dans le projet

Ouvrir le fichier `conversation_service_v3/.env` et ajouter votre clé DeepSeek :

```bash
# DeepSeek API Configuration
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx  # ← Remplacer par votre clé
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### Étape 3 : Exécuter le test de comparaison automatique

#### Option A : Script Python complet (recommandé)

```bash
python test_provider_comparison.py
```

Ce script va :
- ✅ Tester OpenAI avec la question "compare mes dépenses en mai à celle de juin"
- ✅ Tester DeepSeek avec la même question
- ✅ Générer un rapport de comparaison détaillé
- ✅ Comparer latence, qualité, cohérence
- ✅ Sauvegarder les résultats dans un fichier JSON

**Résultat attendu :**
```
🔬 TEST COMPARATIF: OpenAI vs DeepSeek
======================================================================
Question: "compare mes dépenses en mai à celle de juin"
======================================================================

TEST 1/2: OpenAI
...
✅ Response received in X.XXs

TEST 2/2: DeepSeek
...
✅ Response received in X.XXs

📊 RAPPORT DE COMPARAISON
======================================================================
⏱️  LATENCE
  OpenAI:   42.920s
  DeepSeek: X.XXXs

🏆 [Provider] est XX.X% plus rapide

📝 LONGUEUR DES RÉPONSES
  OpenAI:   2908 caractères
  DeepSeek: XXXX caractères

💬 CONTENU DES RÉPONSES
  [Affichage des deux réponses]

📁 Résultats complets sauvegardés dans: test_comparison_results_YYYYMMDD_HHMMSS.json
```

#### Option B : Tests manuels avec curl

**Test 1 : OpenAI** (déjà fait)
```bash
# Configuration
LLM_PROVIDER=openai dans .env

# Redémarrer le service
docker-compose restart conversation_service_v3

# Tester
curl --request POST \
  --url http://localhost:3008/api/v3/conversation/3 \
  --header "Authorization: Bearer [VOTRE_TOKEN]" \
  --header "Content-Type: application/json" \
  --data '{
    "client_info": {"platform": "web", "version": "1.0.0"},
    "message": "compare mes dépenses en mai à celle de juin",
    "message_type": "text",
    "priority": "normal"
  }' \
  -w '\nTime: %{time_total}s\n' \
  > openai_result.json
```

**Test 2 : DeepSeek**
```bash
# Configuration
LLM_PROVIDER=deepseek dans .env
LLM_MODEL=deepseek-chat dans .env
LLM_RESPONSE_MODEL=deepseek-chat dans .env

# Redémarrer le service
docker-compose restart conversation_service_v3

# Attendre 5 secondes
sleep 5

# Tester
curl --request POST \
  --url http://localhost:3008/api/v3/conversation/3 \
  --header "Authorization: Bearer [VOTRE_TOKEN]" \
  --header "Content-Type: application/json" \
  --data '{
    "client_info": {"platform": "web", "version": "1.0.0"},
    "message": "compare mes dépenses en mai à celle de juin",
    "message_type": "text",
    "priority": "normal"
  }' \
  -w '\nTime: %{time_total}s\n' \
  > deepseek_result.json
```

**Comparer les résultats**
```bash
# Voir la latence
echo "OpenAI:" && grep "Time:" openai_result.json
echo "DeepSeek:" && grep "Time:" deepseek_result.json

# Voir la longueur des réponses
echo "OpenAI length:" && cat openai_result.json | jq '.response.message | length'
echo "DeepSeek length:" && cat deepseek_result.json | jq '.response.message | length'

# Comparer les réponses (visuel)
cat openai_result.json | jq '.response.message'
cat deepseek_result.json | jq '.response.message'
```

---

## Métriques à comparer

### 1. Latence ⏱️
- Temps total de traitement
- OpenAI : **42.92s** ← Baseline

### 2. Qualité de la réponse 📊
- Précision des données
- Pertinence de l'analyse
- Personnalisation (utilisation du profil utilisateur)
- Structure et lisibilité

### 3. Cohérence 🎯
- Les chiffres correspondent-ils aux agrégations ?
- Y a-t-il des contradictions ?
- Le ton est-il adapté ?

### 4. Requêtes ES générées 🔍
- **OpenAI a généré :** 2 requêtes (comparative analysis)
  - Période 1 : Mai 2025 (128 transactions)
  - Période 2 : Juin 2025 (134 transactions)
- **DeepSeek génère-t-il les mêmes requêtes ?**
  - Même nombre de queries ?
  - Mêmes filtres ?
  - Mêmes agrégations ?

### 5. Longueur de la réponse 📏
- OpenAI : **2,908 caractères**
- DeepSeek : **? caractères**

---

## Résultats attendus

### Scénario optimiste ✅
- DeepSeek génère les mêmes requêtes ES
- Latence similaire (±20%)
- Qualité comparable
- **→ DeepSeek est viable pour production**

### Scénario réaliste ⚠️
- Requêtes ES légèrement différentes mais correctes
- Latence +10-30% plus lente
- Qualité bonne mais moins de personnalisation
- **→ DeepSeek viable pour dev/test, OpenAI pour production**

### Scénario pessimiste ❌
- Requêtes ES incorrectes ou incomplètes
- Latence >50% plus lente
- Qualité insuffisante
- **→ Rester sur OpenAI uniquement**

---

## Dépannage

### Erreur : "DEEPSEEK_API_KEY not configured"
→ Vérifier que la clé est bien dans `.env` et redémarrer le service

### Erreur : "Invalid API key"
→ Vérifier que la clé commence par `sk-` et est valide sur platform.deepseek.com

### Timeout
→ DeepSeek peut être plus lent, augmenter le timeout dans la requête curl

### Service ne démarre pas
→ Vérifier les logs : `docker logs harena_conversation_v3`

---

## Support

Pour toute question :
1. Consulter `docs/LLM_PROVIDER_GUIDE.md`
2. Voir les logs : `docker logs harena_conversation_v3`
3. Vérifier la configuration : `cat conversation_service_v3/.env | grep -E "LLM_|DEEPSEEK"`

---

**Créé le :** 2025-10-26
**Auteur :** Claude Code
