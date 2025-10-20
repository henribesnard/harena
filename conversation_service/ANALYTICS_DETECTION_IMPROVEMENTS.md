# Améliorations - Détection et Activation de l'Analytics Agent

## 📋 Contexte

L'architecture V1 (conversation_service/) avait déjà :
- ✅ Un `analytics_agent` performant (comparaisons de périodes, tendances, anomalies, pivot tables)
- ✅ Un `reasoning_agent` pour les questions complexes multi-étapes
- ❌ **MAIS** l'analytics_agent n'était jamais activé dans le workflow

**Problème identifié :**
> "Compare mes dépenses de mai à celles de juin" → ne déclenchait PAS l'analytics_agent

---

## 🚀 Solution Implémentée

### 1. Enrichissement du ClassificationResult

**Fichier modifié:** `conversation_service/agents/llm/intent_classifier.py`

Ajout de 3 nouveaux champs dans `ClassificationResult` :

```python
@dataclass
class ClassificationResult:
    # ... champs existants ...

    # NOUVEAUX CHAMPS
    requires_analytics: bool = False
    analytics_type: Optional[str] = None  # "comparison", "trend", "anomaly", "pivot"
    comparison_periods: List[str] = None  # ex: ["2025-05", "2025-06"]
```

### 2. Instructions LLM Enrichies pour Détection d'Analytics

**Ajout au prompt système** (lignes 389-480) :

```
=== DÉTECTION D'ANALYSES AVANCÉES ===

🔍 DÉTECTION DE COMPARAISON (requires_analytics: true, analytics_type: "comparison"):
   - Mots-clés: "compare", "comparer", "comparaison", "vs", "versus", "différence", "variation"
   - Périodes multiples: "mai vs juin", "ce mois vs mois dernier", "2024 vs 2025"
   - Formulations: "entre mai et juin", "du mois d'avril au mois de mai"

   EXEMPLES:
   - "Compare mes dépenses de mai à celles de juin"
     → requires_analytics: true
     → analytics_type: "comparison"
     → comparison_periods: ["2025-05", "2025-06"]

🔍 DÉTECTION DE TENDANCE (requires_analytics: true, analytics_type: "trend"):
   - Mots-clés: "évolution", "progression", "tendance", "trend", "historique sur"

🔍 DÉTECTION D'ANOMALIE (requires_analytics: true, analytics_type: "anomaly"):
   - Mots-clés: "inhabituel", "anormal", "suspect", "étrange", "bizarre"
```

**Format de réponse LLM enrichi :**

```json
{
    "intent_group": "transaction_search",
    "intent_subtype": "comparison",
    "confidence": 0.95,
    "entities": [...],
    "reasoning": "Comparaison des dépenses entre deux mois",
    "requires_analytics": true,
    "analytics_type": "comparison",
    "comparison_periods": ["2025-05", "2025-06"]
}
```

### 3. Extraction dans _build_classification_result()

**Fichier modifié:** `conversation_service/agents/llm/intent_classifier.py` (lignes 580-603)

```python
# NOUVEAU: Extraction des champs analytics
requires_analytics = classification_data.get("requires_analytics", False)
analytics_type = classification_data.get("analytics_type")
comparison_periods = classification_data.get("comparison_periods")

# Log si analytics détecté
if requires_analytics:
    logger.info(f"🔍 Analytics détecté: type={analytics_type}, periods={comparison_periods}")

return ClassificationResult(
    # ... champs existants ...
    requires_analytics=requires_analytics,
    analytics_type=analytics_type,
    comparison_periods=comparison_periods or []
)
```

### 4. Routage dans l'Orchestrateur

**Fichier modifié:** `conversation_service/core/conversation_orchestrator.py` (lignes 239-246)

```python
# === ROUTING: CHECK IF ANALYTICS REQUIRED (NEW PRIORITY) ===
if classification_result.requires_analytics:
    logger.info(f"🔍 Analytics required - type: {classification_result.analytics_type}")

    # TODO: Route to analytics_agent here
    # For now, log and continue to standard pipeline
    logger.warning("Analytics agent routing not yet implemented - using standard pipeline")

# === ROUTING: CHECK IF COMPLEX QUERY NEEDS REASONING AGENT ===
elif self.reasoning_agent and self._is_complex_query(classification_result, request.user_message):
    # ... logique existante ...
```

**Ordre de priorité du routage :**
1. 🆕 **Analytics required** → analytics_agent (TODO: à implémenter)
2. ✅ **Complex query** → reasoning_agent
3. ✅ **Direct response** → standard pipeline
4. ✅ **Search required** → query building + execution + response

---

## 📊 Cas d'Usage Supportés

### Comparaisons de Périodes

| Question | `analytics_type` | `comparison_periods` |
|----------|------------------|----------------------|
| "Compare mes dépenses de mai à celles de juin" | `comparison` | `["2025-05", "2025-06"]` |
| "Différence entre janvier et février" | `comparison` | `["2025-01", "2025-02"]` |
| "Mes achats ce mois vs mois dernier" | `comparison` | `["this_month", "last_month"]` |

### Tendances

| Question | `analytics_type` |
|----------|------------------|
| "Évolution de mes dépenses restaurants sur 6 mois" | `trend` |
| "Tendance de mes achats en ligne" | `trend` |

### Anomalies

| Question | `analytics_type` |
|----------|------------------|
| "Transactions inhabituelles ce mois" | `anomaly` |
| "Dépenses anormales" | `anomaly` |

---

## 🔄 Workflow Complet

```
User Query
    ↓
[Context Analysis]
    ↓
[Intent Classification] ← 🆕 Détecte requires_analytics + analytics_type
    ↓
[Routing Decision]
    ├─→ 🆕 requires_analytics: true → [Analytics Agent] (TODO)
    ├─→ Complex query → [Reasoning Agent]
    ├─→ Direct response → [Response Generator]
    └─→ Search required → [Query Build] → [Query Execute] → [Response Generator]
```

---

## ✅ Avantages

1. **Détection automatique** des besoins d'analytics par le LLM
2. **Séparation des responsabilités** :
   - Intent Classifier → détecte QUOI analyser
   - Analytics Agent → effectue l'analyse (statistiques pures)
   - Reasoning Agent → gère la logique multi-étapes
3. **Extensible** : facile d'ajouter de nouveaux types d'analytics
4. **Zero latence additionnelle** : détection dans l'appel LLM existant
5. **Données structurées** : periods extraites automatiquement

---

## 🚧 TODO - Implémentation Complète de l'Analytics Agent

### Étape suivante : Activer le routage vers analytics_agent

**Dans `conversation_orchestrator.py` (ligne 243-245) :**

```python
if classification_result.requires_analytics:
    logger.info(f"🔍 Analytics required - type: {classification_result.analytics_type}")

    # NOUVEAU CODE À AJOUTER:
    from ..agents.logic.analytics_agent import AnalyticsAgent, PeriodType

    if not hasattr(self, 'analytics_agent'):
        self.analytics_agent = AnalyticsAgent()

    if classification_result.analytics_type == "comparison":
        # Extraire les périodes
        periods = classification_result.comparison_periods

        # Exécuter 2 requêtes pour récupérer les transactions de chaque période
        # period_1_transactions = await query_executor.execute(...)
        # period_2_transactions = await query_executor.execute(...)

        # Comparer avec analytics_agent
        comparison_result = await self.analytics_agent.compare_periods(
            transactions_period_1=period_1_transactions,
            transactions_period_2=period_2_transactions,
            period_type=PeriodType.MOM
        )

        # Générer réponse avec comparison_result
        # response = await response_generator.generate_with_analytics(comparison_result)

    elif classification_result.analytics_type == "trend":
        # trend_result = await self.analytics_agent.calculate_trend(...)
        pass

    elif classification_result.analytics_type == "anomaly":
        # anomalies = await self.analytics_agent.detect_anomalies(...)
        pass
```

### Étape bonus : Enrichir le Response Generator

Créer une méthode `generate_with_analytics()` dans `response_generator.py` pour formater les résultats d'analytics de manière naturelle.

---

## 🧪 Test Recommandé

```python
# Test de détection
question = "Compare mes dépenses de mai à celles de juin"

result = await intent_classifier.classify_intent(
    ClassificationRequest(
        user_message=question,
        conversation_context=[],
        user_id=3
    )
)

assert result.requires_analytics == True
assert result.analytics_type == "comparison"
assert result.comparison_periods == ["2025-05", "2025-06"]
```

---

## 📈 Impact Performance

- **Latence additionnelle** : 0ms (détection dans l'appel LLM existant)
- **Précision** : +XX% sur les questions de comparaison (à mesurer)
- **Cas d'usage couverts** : +3 types d'analytics (comparison, trend, anomaly)

---

## 📝 Notes de Déploiement

1. Les modifications sont **backward-compatible**
2. Si le LLM ne retourne pas `requires_analytics`, la valeur par défaut est `False`
3. Le système continue de fonctionner normalement pour les queries non-analytics
4. Les logs incluent 🔍 pour faciliter le debugging des analytics

---

**Date:** 2025-10-20
**Version:** v1.1 - Analytics Detection Enhancement
