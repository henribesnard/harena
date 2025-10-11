# Analyse Approfondie du Workflow Harena - Architecture v2.0

**Date**: 2025-01-11
**Version**: 2.0 (Phase 5 complète)
**Auteur**: Analyse Architecturale Expert

---

## Table des Matières

1. [Vue d'Ensemble Architecture Actuelle](#1-vue-densemble-architecture-actuelle)
2. [Workflow Détaillé Pipeline Conversation](#2-workflow-détaillé-pipeline-conversation)
3. [Capacités et Limitations Actuelles](#3-capacités-et-limitations-actuelles)
4. [Recommandations d'Amélioration pour Analyses Complexes](#4-recommandations-damélioration-pour-analyses-complexes)
5. [Plan d'Implémentation par Phases](#5-plan-dimplémentation-par-phases)

---

## 1. Vue d'Ensemble Architecture Actuelle

### 1.1 Architecture Microservices

L'architecture Harena repose sur 6 services principaux :

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway (CloudFront)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
      ┌─────────▼─────────┐          ┌─────────▼──────────┐
      │  User Service     │          │ Conversation        │
      │  (Auth, Profiles) │          │ Service (IA Core)   │
      └───────────────────┘          └─────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────┐
                    │                          │                  │
          ┌─────────▼────────┐      ┌─────────▼────────┐  ┌─────▼──────┐
          │  Search Service  │      │  Metric Service  │  │ Sync       │
          │  (OpenSearch)    │      │  (Analytics)     │  │ Service    │
          └──────────────────┘      └──────────────────┘  └────────────┘
                    │
          ┌─────────▼────────┐
          │  Enrichment      │
          │  Service         │
          └──────────────────┘
```

### 1.2 Architecture Pipeline Conversation (v2.0)

**Philosophie**: Pipeline linéaire avec agents spécialisés (logiques + LLM)

```
User Input
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│         CONVERSATION ORCHESTRATOR (Coordination)               │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
[STAGE 1] Context Manager (Logique)
    │   - Récupération historique conversation
    │   - Compression contexte si > 8000 tokens
    │   - Extraction préférences utilisateur
    │
    ▼
[STAGE 2] Intent Classifier (LLM - DeepSeek)
    │   - Classification intention (few-shot prompting)
    │   - Extraction entités (montants, dates, marchands...)
    │   - Scores de confiance
    │
    ▼
[STAGE 3] Query Builder (Logique)
    │   - Mapping intention → template query
    │   - Injection entités dans template
    │   - Validation query Elasticsearch
    │
    ▼
[STAGE 4] Query Executor (Logique)
    │   - Exécution requête search_service
    │   - Récupération automatique de toutes les pages
    │   - Cache résultats (TTL 5min)
    │   - Retry automatique (2x)
    │
    ▼
[STAGE 5] Response Generator (LLM - DeepSeek)
    │   - Génération réponse personnalisée
    │   - Insights automatiques (patterns, anomalies)
    │   - Support streaming temps réel
    │
    ▼
User Response (JSON structuré)
```

### 1.3 Technologies Clés

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **LLM** | DeepSeek Chat (128K tokens) | Classification intent + génération réponse |
| **Search** | OpenSearch/Elasticsearch | Indexation transactions + recherche |
| **Storage** | PostgreSQL (RDS) | Données structurées (users, conversations) |
| **Cache** | Redis | Cache sémantique + sessions |
| **Frontend** | CloudFront + S3 | Distribution statique globale |
| **API Gateway** | ALB + CloudFront | Routage + HTTPS + CORS |

---

## 2. Workflow Détaillé Pipeline Conversation

### 2.1 STAGE 1: Context Analysis (Logique)

**Fichier**: `conversation_service/core/context_manager.py`

**Responsabilités**:
- Récupération du contexte de conversation depuis PostgreSQL
- Compression intelligente si tokens > 8000 (préservation des informations critiques)
- Extraction des intentions récentes (3 derniers tours)
- Construction du profil utilisateur (préférences, habitudes)

**Métriques**:
- Temps moyen: ~50ms
- Compression ratio: ~40-60% si activée
- Cache hit rate: N/A (toujours récupération DB)

**Limitations**:
- ❌ Pas de mémorisation long-terme des préférences
- ❌ Compression basique (pas de clustering sémantique)
- ❌ Pas d'analyse du sentiment ou du ton utilisateur

### 2.2 STAGE 2: Intent Classification (LLM)

**Fichier**: `conversation_service/agents/llm/intent_classifier.py`

**Responsabilités**:
- Classification intention utilisateur (14 intents supportés)
- Extraction entités (montants, dates, marchands, catégories, opérations)
- Few-shot learning avec 15+ exemples contextualisés
- Normalisation automatique (fautes de frappe, formats dates)

**Capacités d'Extraction**:

| Type Entité | Format Supporté | Exemples |
|-------------|-----------------|----------|
| **Montants** | Simple, plage, opérateurs (gt, lt, gte, lte, eq, range) | "plus de 100€", "entre 50 et 100€" |
| **Dates** | Relatives, absolues, plages, mois, années | "ce mois", "5 mars", "2024-05", "du 1er au 15 mai" |
| **Marchands** | Noms exacts, liste, corrections fautes | "Amazon", "Tesla", "Amazon Netflix Disney+" |
| **Catégories** | Catégories dynamiques DB | "Alimentation", "Transport", "Loisirs" |
| **Opérations** | 6 types | card, withdrawal, direct_debit, transfer, deferred_debit_card, unknown |
| **Transaction Type** | 3 types | debit, credit, all |

**Métriques**:
- Temps moyen: ~800-1200ms (appel DeepSeek API)
- Confidence moyenne: 0.87
- Taux high confidence (>0.8): 75%
- Fallback utilisé: <5%

**Limitations**:
- ❌ Pas de compréhension des questions complexes multi-étapes
- ❌ Pas de mémoire des clarifications précédentes
- ❌ Extraction limitée à des patterns connus (pas de raisonnement sur nouveaux concepts)
- ❌ Pas de détection d'intentions imbriquées ("Mes achats Amazon de plus de 50€ du mois dernier")

### 2.3 STAGE 3: Query Building (Logique)

**Fichier**: `conversation_service/core/query_builder.py`

**Responsabilités**:
- Sélection template query basé sur intention + sous-type
- Injection des entités extraites dans le template
- Normalisation format Elasticsearch
- Validation schema query

**Templates Disponibles**:
- `transaction_search.simple`: Recherche basique toutes transactions
- `transaction_search.filter`: Filtres multiples (montant, date, marchand, catégorie)
- `transaction_search.aggregate`: Agrégations (sum, avg, group by)
- `financial_query.balance`: Solde compte
- `financial_query.expenses`: Total dépenses

**Métriques**:
- Temps moyen: ~10-20ms (pure logique)
- Taux succès validation: >95%
- Templates utilisés: 12 templates actifs

**Limitations**:
- ❌ Pas de requêtes dynamiques (templates figés)
- ❌ Pas de jointures multi-entités (ex: transactions + catégories + marchands)
- ❌ Pas de requêtes ML/scoring avancé (similarité, clustering)
- ❌ Pas de requêtes temporelles complexes (rolling windows, trends)

### 2.4 STAGE 4: Query Execution (Logique)

**Fichier**: `conversation_service/core/query_executor.py`

**Responsabilités**:
- Exécution requête HTTP vers `search_service`
- Récupération automatique de TOUTES les pages de résultats
- Retry automatique (2 tentatives avec backoff exponentiel)
- Cache résultats (TTL 5 minutes)
- Gestion timeouts et erreurs HTTP

**Optimisations**:
- Pagination automatique jusqu'à 100 pages max (protection boucles infinies)
- Connection pooling (limite 50 connexions concurrentes)
- Cache intelligent avec clés hash (query + user_id)

**Métriques**:
- Temps moyen: ~200-500ms (selon taille résultats)
- Cache hit rate: ~30%
- Retry rate: <5%
- Timeout rate: <1%

**Limitations**:
- ❌ Pas de streaming des résultats (charge mémoire si >1000 transactions)
- ❌ Pas d'agrégations post-traitement (tout doit être dans la query ES)
- ❌ Pas de parallélisation multi-requêtes
- ❌ Cache non distribué (mémoire locale)

### 2.5 STAGE 5: Response Generation (LLM)

**Fichier**: `conversation_service/agents/llm/response_generator.py`

**Responsabilités**:
- Génération réponse personnalisée en langage naturel
- Insights automatiques (7 types)
- Filtrage données pour optimisation tokens (14 champs → 6 champs essentiels)
- Support streaming temps réel (WebSocket)
- Templates de réponse par intention

**Insights Automatiques**:

| Type Insight | Déclencheur | Utilité |
|--------------|-------------|---------|
| `spending_pattern` | Dépenses >20% au-dessus moyenne | Alerte dépenses inhabituelles |
| `income_pattern` | Revenus >50% au-dessus moyenne | Notification revenus exceptionnels |
| `unusual_transaction` | Transaction >2x moyenne | Détection anomalies |
| `budget_alert` | >80% budget mensuel | Alerte dépassement budget |
| `trend_analysis` | Analyse historique | Tendances temporelles |
| `recommendation` | Patterns récurrents | Conseils personnalisés |
| `financial_summary` | Agrégations | Résumé débits/crédits |

**Optimisations Tokens**:
- Filtrage transactions: 14 champs → 6 champs essentiels (~57% réduction)
- Limite transactions: max 1500 (budget 80K tokens)
- Agrégations compactes: top 5 marchands only
- Contexte conversation: 2 derniers tours uniquement

**Métriques**:
- Temps moyen: ~1000-1500ms (appel DeepSeek API)
- Tokens moyens utilisés: ~3000-5000 tokens
- Insights générés: 1-3 par réponse
- Taux streaming: 15% des requêtes

**Limitations**:
- ❌ Pas de génération graphiques/visualisations avancées
- ❌ Pas de prédictions ML (tendances futures)
- ❌ Pas de recommandations actionnables (ex: "Créer une règle d'alerte")
- ❌ Insights statiques (pas d'apprentissage des préférences)

---

## 3. Capacités et Limitations Actuelles

### 3.1 Points Forts

✅ **Pipeline Robuste et Scalable**
- Architecture microservices découplée
- Retry automatique et gestion erreurs
- Cache multi-niveaux (Redis + mémoire)
- Support streaming temps réel

✅ **Classification Intent Puissante**
- Few-shot learning performant
- Extraction entités complexes (dates, montants, marchands)
- Normalisation automatique (fautes, formats)
- Fallback intelligent si LLM échoue

✅ **Execution Query Optimisée**
- Récupération automatique toutes pages
- Connection pooling
- Cache résultats avec TTL
- Pagination transparente

✅ **Génération Réponse Contextuelle**
- Insights automatiques pertinents
- Personnalisation ton/style
- Support streaming
- Optimisation tokens (filtrage champs)

### 3.2 Limitations Majeures pour Analyses Complexes

#### 3.2.1 Compréhension Questions Complexes

**Problème**: Le système gère mal les questions multi-étapes ou nécessitant un raisonnement complexe.

**Exemples non supportés**:
```
❌ "Quel est le pourcentage d'augmentation de mes dépenses alimentaires
    par rapport au même mois l'année dernière ?"

    → Nécessite: Agrégation mois M année N, agrégation mois M année N-1,
                calcul % variation, contexte temporel année-over-année

❌ "Parmi mes abonnements de plus de 10€, lesquels ont augmenté de prix
    ces 6 derniers mois ?"

    → Nécessite: Détection récurrence, tracking variations prix,
                analyse historique 6 mois, jointure temporelle

❌ "Compare mes dépenses restaurant avec la moyenne de mes amis
    dans ma catégorie d'âge"

    → Nécessite: Données externes (benchmarks), agrégation multi-users,
                segmentation démographique, calcul moyenne cohort

❌ "Prédis mes dépenses pour le mois prochain en tenant compte
    de mes patterns d'achat"

    → Nécessite: Modèle prédictif ML, analyse patterns saisonniers,
                calcul tendances, forecast multi-catégories
```

**Cause racine**:
- Intent classifier ne décompose pas les questions complexes en sous-tâches
- Query builder limité à des templates statiques (pas de composition de queries)
- Pas de moteur de raisonnement pour orchestrer plusieurs requêtes séquentielles
- Pas de mémoire inter-requêtes pour contexte étendu

#### 3.2.2 Analyses Temporelles Avancées

**Problème**: Les analyses YoY, MoM, trends, saisonnalité sont limitées.

**Capacités actuelles**:
- ✅ Filtrage par mois spécifique: "Mes dépenses de mai"
- ✅ Plages de dates fixes: "Du 1er au 15 mai"
- ❌ Comparaisons temporelles: "Mai 2024 vs Mai 2025"
- ❌ Agrégations rolling: "Moyenne mobile 3 mois"
- ❌ Détection saisonnalité: "Mois où je dépense le plus"
- ❌ Trends: "Evolution dépenses 12 derniers mois"

**Cause racine**:
- Templates query ne supportent qu'une seule période
- Pas d'agrégations temporelles dans Elasticsearch (date_histogram)
- Response generator ne sait pas calculer des métriques comparatives
- Pas de stockage pré-calculé des métriques YoY/MoM

#### 3.2.3 Recommandations Actionnables

**Problème**: Les insights sont informatifs mais pas actionnables.

**Insights actuels**:
```
✅ "Vos dépenses récentes sont 25% au-dessus de votre moyenne"
   → Informatif, mais pas actionnable

❌ Recommandation attendue:
   "Vos dépenses alimentaires ont augmenté de 25%.
    Créez une alerte à 150€/mois pour suivre cette catégorie ?"
   → Actionnable avec bouton CTA
```

**Cas d'usage manquants**:
- ❌ Créer règles d'alerte ("M'alerter si dépense >100€")
- ❌ Optimisation abonnements ("Netflix détecté, voulez-vous partager ?")
- ❌ Détection doublons ("2 abonnements Spotify détectés")
- ❌ Opportunités économies ("Vous pourriez économiser 50€/mois en changeant de forfait")

**Cause racine**:
- Pas d'intégration avec système de règles/alertes
- Pas de base de connaissances pour optimisations
- Insights non connectés à des actions dans l'UI
- Pas de tracking efficacité recommandations

#### 3.2.4 Mémoire et Apprentissage

**Problème**: Pas de mémorisation long-terme des préférences et habitudes utilisateur.

**Exemples**:
```
❌ L'utilisateur demande régulièrement "Mes dépenses Tesla"
   → Le système devrait suggérer: "Créer un raccourci 'Mes achats Tesla' ?"

❌ L'utilisateur pose toujours des questions sur l'alimentation
   → Le système devrait adapter: "Voulez-vous un rapport mensuel automatique ?"

❌ L'utilisateur demande systématiquement des comparaisons YoY
   → Le système devrait suggérer: "Activer analyse YoY automatique ?"
```

**Cause racine**:
- Context manager ne stocke que l'historique conversation (court-terme)
- Pas de profil utilisateur évolutif (long-terme)
- Pas d'analyse patterns de requêtes
- Pas de système de préférences explicites/implicites

#### 3.2.5 Analyses Multi-Catégories et Croisées

**Problème**: Difficile d'analyser plusieurs dimensions simultanément.

**Exemples non supportés**:
```
❌ "Quelle est la répartition de mes dépenses par catégorie
    pour chaque mois des 6 derniers mois ?"

    → Nécessite: Agrégation 2D (catégories × mois), pivot table

❌ "Quels marchands représentent le plus de dépenses dans
    la catégorie Transport ?"

    → Nécessite: Agrégation imbriquée (filter catégorie → group by marchand)

❌ "Compare mes dépenses en ligne vs en magasin physique
    par catégorie"

    → Nécessite: Enrichissement données (en ligne vs physique),
                agrégation multi-dimensions
```

**Cause racine**:
- Templates query limités à 1 dimension d'agrégation
- Pas de support pivot tables / data cubes
- Response generator ne sait pas présenter données multi-dimensionnelles
- Pas de visualisations graphiques avancées (heatmaps, sunburst)

---

## 4. Recommandations d'Amélioration pour Analyses Complexes

### 4.1 Architecture Multi-Agent Avancée

#### 4.1.1 Ajouter un "Reasoning Agent" (Orchestration Intelligente)

**Objectif**: Décomposer les questions complexes en sous-tâches exécutables.

**Architecture proposée**:
```
User Input: "Mes dépenses alimentaires ce mois vs mois dernier"
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  REASONING AGENT (Nouveau - LLM avec Chain-of-Thought)     │
│  - Décomposition question en étapes                        │
│  - Plan d'exécution séquentiel/parallèle                   │
│  - Gestion état inter-étapes                               │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Plan Généré:                                    │
│  1. Récupérer dépenses alimentaires ce mois      │
│  2. Récupérer dépenses alimentaires mois dernier │
│  3. Calculer delta et % variation                │
│  4. Identifier top 3 contributeurs augmentation  │
└──────────────────────────────────────────────────┘
    │
    ▼
Exécution parallèle des étapes 1 et 2 via Query Executor
    │
    ▼
Calcul métrique en Python (étape 3)
    │
    ▼
Response Generator avec contexte enrichi
```

**Technologies**:
- LLM: DeepSeek ou Claude (meilleur raisonnement) avec prompt Chain-of-Thought
- Framework: LangGraph ou AutoGen pour orchestration multi-agents
- Storage: Redis pour état temporaire entre étapes

**Bénéfices**:
- ✅ Support questions multi-étapes
- ✅ Gestion erreurs granulaire par étape
- ✅ Explicabilité du raisonnement (traçabilité plan)
- ✅ Réutilisation sous-tâches (cache)

**Effort estimé**: 3-4 semaines (2 devs)

#### 4.1.2 Créer un "Analytics Agent" (Métriques Avancées)

**Objectif**: Calculs statistiques et comparaisons temporelles avancées.

**Capacités**:
- Comparaisons YoY, MoM, YTD, QoQ
- Moyennes mobiles (rolling averages)
- Détection anomalies statistiques (z-score, IQR)
- Calcul trends (régression linéaire)
- Agrégations multi-dimensionnelles (pivot)

**Architecture proposée**:
```python
class AnalyticsAgent:
    async def compare_periods(
        self,
        transactions_period_1: List[Dict],
        transactions_period_2: List[Dict],
        metrics: List[str] = ["sum", "avg", "count"]
    ) -> ComparisonResult:
        """Compare deux périodes avec calculs delta et %"""

    async def detect_anomalies(
        self,
        transactions: List[Dict],
        method: str = "zscore",  # zscore, iqr, isolation_forest
        threshold: float = 2.0
    ) -> List[Anomaly]:
        """Détecte transactions anormales"""

    async def calculate_trend(
        self,
        transactions: List[Dict],
        aggregation: str = "daily",  # daily, weekly, monthly
        forecast_periods: int = 3
    ) -> TrendAnalysis:
        """Calcule tendance avec forecast simple"""

    async def pivot_analysis(
        self,
        transactions: List[Dict],
        rows: str = "category",
        columns: str = "month",
        values: str = "amount",
        aggfunc: str = "sum"
    ) -> PivotTable:
        """Génère table pivot multi-dimensionnelle"""
```

**Technologies**:
- Python: pandas, numpy, scipy pour calculs
- Caching: Redis pour résultats pré-calculés (TTL 1h)
- Monitoring: métriques performance calculs

**Bénéfices**:
- ✅ Analyses temporelles complexes (YoY, MoM)
- ✅ Détection anomalies automatique
- ✅ Trends et forecasts simples
- ✅ Analyses multi-dimensionnelles

**Effort estimé**: 2-3 semaines (1 dev)

#### 4.1.3 Créer un "Recommendation Engine" (Insights Actionnables)

**Objectif**: Générer des recommandations personnalisées et actionnables.

**Types de recommandations**:

| Catégorie | Règle de Détection | Action Proposée |
|-----------|-------------------|-----------------|
| **Optimisation Abonnements** | 2+ abonnements streaming détectés | "Consolidez vos abonnements → économie 15€/mois" |
| **Doublons** | Même marchand 2x même jour | "Transaction dupliquée détectée ?" |
| **Alertes Budget** | >80% budget catégorie | "Créer alerte à 90% pour Alimentation" |
| **Opportunités Cashback** | Achats Amazon >100€/mois | "Activez carte cashback 3% Amazon" |
| **Patterns Inattendus** | Dépenses loisirs +50% vs avg | "Vos loisirs ont augmenté, budget à ajuster ?" |

**Architecture proposée**:
```python
class RecommendationEngine:
    def __init__(self, rule_engine: RuleEngine, user_profile: UserProfile):
        self.rules = self._load_recommendation_rules()
        self.user_profile = user_profile

    async def generate_recommendations(
        self,
        transactions: List[Dict],
        user_context: Dict[str, Any],
        max_recommendations: int = 3
    ) -> List[Recommendation]:
        """Génère top N recommandations actionnables"""

        recommendations = []

        # Évaluer toutes les règles
        for rule in self.rules:
            if rule.evaluate(transactions, user_context):
                recommendation = rule.build_recommendation()
                recommendation.score = self._calculate_relevance_score(
                    recommendation, user_profile
                )
                recommendations.append(recommendation)

        # Trier par score pertinence
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations[:max_recommendations]

    def _calculate_relevance_score(
        self,
        recommendation: Recommendation,
        user_profile: UserProfile
    ) -> float:
        """Score basé sur historique utilisateur et impact estimé"""

        # Facteurs:
        # - Impact financier estimé (€)
        # - Pertinence historique (a-t-il déjà accepté ce type ?)
        # - Urgence (risque dépassement budget)
        # - Facilité d'implémentation

        return score
```

**Base de Règles** (stockée en DB/config):
```yaml
recommendations:
  - id: "optimize_subscriptions"
    name: "Optimisation Abonnements"
    condition:
      type: "multiple_subscriptions"
      category: "streaming"
      min_count: 2
    action:
      type: "suggest_consolidation"
      estimated_savings: "auto_calculate"
      cta: "Voir les options"

  - id: "budget_alert_setup"
    name: "Créer Alerte Budget"
    condition:
      type: "budget_threshold"
      percentage: 0.8
    action:
      type: "create_alert"
      threshold: 0.9
      cta: "Activer l'alerte"
```

**Bénéfices**:
- ✅ Recommandations personnalisées et actionnables
- ✅ Intégration UI/UX (boutons CTA)
- ✅ Tracking efficacité (A/B testing)
- ✅ Apprentissage préférences utilisateur

**Effort estimé**: 3-4 semaines (1-2 devs)

### 4.2 Enrichissement Données et Contexte

#### 4.2.1 User Profile Évolutif (Long-Term Memory)

**Objectif**: Mémoriser préférences et habitudes utilisateur sur le long terme.

**Structure proposée**:
```python
class UserProfile:
    user_id: int

    # Préférences explicites
    preferred_categories: List[str]  # ["Alimentation", "Transport"]
    preferred_merchants: List[str]   # ["Amazon", "Tesla"]
    notification_preferences: Dict

    # Habitudes implicites (ML)
    frequent_query_patterns: List[str]  # ["YoY comparisons", "monthly reports"]
    average_spending_by_category: Dict[str, float]
    peak_spending_days: List[str]  # ["Monday", "Friday"]

    # Historique interactions
    accepted_recommendations: List[str]
    dismissed_recommendations: List[str]
    created_alerts: List[Alert]

    # Métadonnées
    created_at: datetime
    last_updated: datetime
    profile_completeness: float  # 0.0 to 1.0
```

**Mise à jour automatique**:
- Analyse patterns queries toutes les semaines (cron job)
- Recalcul moyennes dépenses tous les mois
- Inférence préférences depuis actions utilisateur (recommandations acceptées)

**Bénéfices**:
- ✅ Personnalisation proactive
- ✅ Suggestions de raccourcis
- ✅ Rapports automatiques personnalisés
- ✅ Réduction friction utilisateur

**Effort estimé**: 2 semaines (1 dev)

#### 4.2.2 Pre-Computed Metrics (Optimisation Performance)

**Objectif**: Pré-calculer métriques complexes pour réponses instantanées.

**Métriques à pré-calculer** (batch nightly):
```python
class PrecomputedMetrics:
    user_id: int
    period: str  # "2025-01", "2025-Q1", "2025"

    # Totaux
    total_debit: float
    total_credit: float
    net_cashflow: float

    # Par catégorie
    spending_by_category: Dict[str, float]
    top_5_categories: List[Tuple[str, float]]

    # Par marchand
    spending_by_merchant: Dict[str, float]
    top_10_merchants: List[Tuple[str, float]]

    # Comparaisons temporelles
    mom_change: float  # Month-over-Month %
    yoy_change: float  # Year-over-Year %

    # Détections
    unusual_transactions: List[int]  # transaction_ids
    recurring_subscriptions: List[Dict]

    # Métadonnées
    computed_at: datetime
    transactions_count: int
```

**Architecture**:
```
Nightly Batch Job (23:00 UTC)
    │
    ▼
┌────────────────────────────────────────────┐
│  Metrics Computation Engine                │
│  - Récupère transactions du mois           │
│  - Calcule métriques agrégées              │
│  - Détecte anomalies et patterns           │
│  - Stocke résultats dans Redis + DB        │
└────────────────────────────────────────────┘
    │
    ▼
Redis Cache (TTL 24h) + PostgreSQL (historique)
    │
    ▼
Conversation Service (lecture instantanée)
```

**Bénéfices**:
- ✅ Réponses instantanées pour métriques complexes (<50ms)
- ✅ Réduction charge LLM (moins de calculs runtime)
- ✅ Historique métriques pour analyses long-terme
- ✅ Détection proactive anomalies

**Effort estimé**: 2-3 semaines (1 dev)

### 4.3 Amélioration Query Building et Execution

#### 4.3.1 Dynamic Query Composition (Au-delà des Templates)

**Objectif**: Construire des queries Elasticsearch dynamiques et complexes.

**Architecture actuelle** (limitée):
```python
# Template statique
TEMPLATES = {
    "transaction_search.filter": {
        "filters": {
            "merchant_name": "{{merchant_name}}",
            "amount": "{{montant}}"
        }
    }
}
```

**Architecture proposée** (dynamique):
```python
class DynamicQueryBuilder:
    def __init__(self, es_schema: ElasticsearchSchema):
        self.schema = es_schema

    async def build_query(
        self,
        intent: str,
        entities: Dict[str, Any],
        operations: List[QueryOperation]
    ) -> Dict[str, Any]:
        """Construit query ES dynamique avec composition d'opérations"""

        query = {"query": {"bool": {"must": [], "filter": [], "should": []}}}

        # Composition dynamique des clauses
        for operation in operations:
            if operation.type == "filter":
                query["query"]["bool"]["filter"].append(
                    self._build_filter_clause(operation.field, operation.value)
                )
            elif operation.type == "aggregate":
                query["aggs"] = query.get("aggs", {})
                query["aggs"][operation.name] = self._build_aggregation(
                    operation.agg_type, operation.field
                )
            elif operation.type == "sort":
                query["sort"] = [{operation.field: operation.order}]

        # Validation schema Elasticsearch
        self._validate_query(query)

        return query

    def _build_aggregation(
        self,
        agg_type: str,  # "sum", "avg", "date_histogram", "terms"
        field: str,
        params: Dict = None
    ) -> Dict[str, Any]:
        """Construit agrégation ES dynamique"""

        if agg_type == "date_histogram":
            return {
                "date_histogram": {
                    "field": field,
                    "calendar_interval": params.get("interval", "month")
                }
            }
        elif agg_type == "terms":
            return {
                "terms": {
                    "field": field,
                    "size": params.get("size", 10)
                }
            }
        # ... autres types
```

**Support nouvelles capacités**:
- ✅ Agrégations imbriquées (nested aggregations)
- ✅ Pipelines d'agrégation (bucket_script, cumulative_sum)
- ✅ Jointures via parent-child ou nested documents
- ✅ Scoring personnalisé (function_score)

**Bénéfices**:
- ✅ Queries multi-dimensionnelles (catégories × mois)
- ✅ Analyses temporelles avancées (date_histogram)
- ✅ Agrégations calculées (delta, ratios)
- ✅ Flexibilité illimitée

**Effort estimé**: 3-4 semaines (1 dev)

#### 4.3.2 Query Optimization et Caching Avancé

**Objectif**: Optimiser performance queries lentes et améliorer cache hit rate.

**Stratégies**:

1. **Query Planning Intelligent**:
```python
class QueryOptimizer:
    async def optimize_query(self, query: Dict) -> Dict:
        """Optimise query avant exécution"""

        # Réordonnancement clauses (plus sélectives d'abord)
        query = self._reorder_filters(query)

        # Ajout hints Elasticsearch
        query["_source"] = self._select_minimal_fields(query)

        # Utilisation index optimaux
        query["index"] = self._select_best_index(query)

        return query
```

2. **Caching Multi-Niveau**:
```
L1: Mémoire locale (LRU Cache, 100 queries, TTL 5min)
    │
    ▼ (miss)
L2: Redis distribué (1000 queries, TTL 1h)
    │
    ▼ (miss)
L3: Elasticsearch (source of truth)
```

3. **Pre-fetching Prédictif**:
- Analyser patterns queries utilisateur
- Pré-charger queries fréquentes en arrière-plan
- Cache warm-up au démarrage

**Bénéfices**:
- ✅ Réduction latence P95: 500ms → 150ms
- ✅ Cache hit rate: 30% → 70%
- ✅ Réduction charge Elasticsearch
- ✅ Scaling horizontal (Redis distribué)

**Effort estimé**: 2 semaines (1 dev)

### 4.4 Response Generation Avancée

#### 4.4.1 Visualisations Graphiques Natives

**Objectif**: Générer des visualisations riches (graphiques, tables) dans la réponse.

**Types de visualisations**:

| Type | Use Case | Format |
|------|----------|--------|
| **Line Chart** | Evolution temporelle dépenses | Chart.js / Recharts |
| **Bar Chart** | Comparaison catégories | Chart.js / Recharts |
| **Pie Chart** | Répartition % par catégorie | Chart.js / Recharts |
| **Heatmap** | Dépenses par jour de semaine × heure | D3.js |
| **Sunburst** | Hiérarchie catégories (groupe → sous-catégorie) | D3.js |
| **Table** | Liste transactions détaillées | React Table |
| **KPI Cards** | Métriques clés (total, moyenne, trend) | Custom Component |

**Architecture proposée**:
```python
class VisualizationGenerator:
    async def generate_visualizations(
        self,
        data: List[Dict],
        intent: str,
        visualization_types: List[str] = None
    ) -> List[Visualization]:
        """Génère visualisations adaptées aux données et intention"""

        visualizations = []

        # Détection automatique du meilleur type de viz
        if not visualization_types:
            visualization_types = self._auto_detect_viz_types(data, intent)

        for viz_type in visualization_types:
            if viz_type == "line_chart":
                viz = self._create_line_chart(data)
            elif viz_type == "bar_chart":
                viz = self._create_bar_chart(data)
            elif viz_type == "heatmap":
                viz = self._create_heatmap(data)
            # ...

            visualizations.append(viz)

        return visualizations

    def _auto_detect_viz_types(
        self,
        data: List[Dict],
        intent: str
    ) -> List[str]:
        """Détecte automatiquement les visualisations pertinentes"""

        # Règles heuristiques
        if intent == "transaction_search.by_period":
            return ["line_chart", "kpi_cards"]
        elif intent == "financial_query.expenses":
            return ["pie_chart", "bar_chart"]
        elif "comparison" in intent:
            return ["bar_chart", "table"]
        # ...
```

**Format réponse**:
```json
{
  "response_text": "Voici vos dépenses du mois...",
  "visualizations": [
    {
      "type": "line_chart",
      "title": "Evolution Dépenses Janvier",
      "data": {
        "labels": ["01/01", "02/01", ...],
        "datasets": [{"label": "Dépenses", "data": [120, 85, ...]}]
      },
      "options": {"responsive": true, "scales": {...}}
    },
    {
      "type": "kpi_cards",
      "cards": [
        {"label": "Total", "value": "1,234€", "trend": "+12%"},
        {"label": "Moyenne", "value": "41€/jour", "trend": "-5%"}
      ]
    }
  ]
}
```

**Bénéfices**:
- ✅ Compréhension rapide données visuelles
- ✅ Détection patterns visuels (pics, creux)
- ✅ Expérience utilisateur moderne
- ✅ Export/partage graphiques

**Effort estimé**: 3-4 semaines (1 dev frontend + 1 dev backend)

#### 4.4.2 Explainability et Transparence

**Objectif**: Expliquer le raisonnement du système à l'utilisateur.

**Niveaux d'explicabilité**:

1. **Pipeline Stage Tracing**:
```json
{
  "response": "Vous avez 15 transactions Tesla...",
  "reasoning": {
    "stage_1_intent": {
      "detected": "transaction_search.by_merchant",
      "confidence": 0.92,
      "entities_extracted": ["merchant: Tesla", "transaction_type: debit"]
    },
    "stage_2_query": {
      "filters_applied": ["merchant_name: Tesla", "transaction_type: debit"],
      "aggregations": ["sum_amount", "count"]
    },
    "stage_3_execution": {
      "results_found": 15,
      "total_amount": 3450.00,
      "time_ms": 234
    },
    "stage_4_insights": {
      "generated": ["unusual_transaction"],
      "reasoning": "Transaction de 1200€ est 3x votre moyenne"
    }
  }
}
```

2. **Why This Answer?**:
- Bouton UI "Pourquoi cette réponse ?" qui affiche le raisonnement
- Tooltip sur chaque insight expliquant la logique de détection

3. **Data Provenance**:
- Traçabilité sources de données (quelle transaction contribue à quel insight)
- Liens cliquables vers transactions sources

**Bénéfices**:
- ✅ Confiance utilisateur accrue
- ✅ Debugging facile pour support
- ✅ Transparence IA
- ✅ Formation utilisateur

**Effort estimé**: 2 semaines (1 dev)

### 4.5 Scalabilité et Performance

#### 4.5.1 Async Processing pour Queries Lentes

**Objectif**: Éviter timeouts sur analyses complexes (>30s).

**Architecture proposée**:
```
User Request (complex query)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Conversation Orchestrator                   │
│  - Détecte query complexe (>10s estimé)      │
│  - Crée job async avec job_id                │
│  - Retourne immédiatement: "En cours..."     │
└──────────────────────────────────────────────┘
    │
    │ (async)
    ▼
┌──────────────────────────────────────────────┐
│  Background Worker (Celery / RQ)             │
│  - Execute query complexe                    │
│  - Calcule métriques                         │
│  - Génère insights                           │
│  - Stocke résultat en Redis avec job_id      │
└──────────────────────────────────────────────┘
    │
    │ (webhook ou polling)
    ▼
Frontend récupère résultat avec job_id
```

**Format réponse initiale**:
```json
{
  "status": "processing",
  "job_id": "job_12345",
  "estimated_time_seconds": 45,
  "message": "Votre analyse est en cours, cela peut prendre jusqu'à 1 minute..."
}
```

**Polling endpoint**:
```
GET /api/v1/conversation/job/{job_id}

Response:
{
  "status": "completed",  // ou "processing", "failed"
  "progress": 100,
  "result": { ... },
  "processing_time_ms": 42000
}
```

**Bénéfices**:
- ✅ Pas de timeout utilisateur
- ✅ Support analyses ultra-complexes (plusieurs minutes)
- ✅ Feedback temps réel (progress bar)
- ✅ Retry automatique si échec

**Effort estimé**: 2-3 semaines (1 dev)

#### 4.5.2 Distributed Caching (Redis Cluster)

**Objectif**: Scaler cache horizontalement pour millions d'utilisateurs.

**Architecture actuelle** (limitée):
- Cache mémoire local (conversation_service)
- Pas de partage entre instances
- Limite mémoire par instance

**Architecture proposée** (scalable):
```
┌─────────────────────────────────────────────────────────┐
│           Redis Cluster (3 masters + 3 replicas)        │
│  - Sharding automatique par clé (user_id hash)          │
│  - Réplication master-replica pour HA                   │
│  - Eviction policy: LRU (Least Recently Used)           │
└─────────────────────────────────────────────────────────┘
            ▲              ▲              ▲
            │              │              │
    ┌───────┴─────┐  ┌────┴────┐  ┌─────┴──────┐
    │ Conv Svc #1 │  │ Conv #2 │  │ Conv #3    │
    │ (Pod)       │  │ (Pod)   │  │ (Pod)      │
    └─────────────┘  └─────────┘  └────────────┘
```

**Stratégie clés de cache**:
```python
# User profile
key = f"user_profile:{user_id}"
ttl = 3600  # 1 heure

# Query results
key = f"query_result:{hash(query)}:{user_id}"
ttl = 300  # 5 minutes

# Pre-computed metrics
key = f"metrics:{user_id}:{period}"  # ex: "metrics:100:2025-01"
ttl = 86400  # 24 heures

# Insights
key = f"insights:{user_id}:{hash(context)}"
ttl = 1800  # 30 minutes
```

**Bénéfices**:
- ✅ Scaling horizontal illimité
- ✅ Haute disponibilité (failover automatique)
- ✅ Cache partagé entre instances
- ✅ Réduction latence P99: 1000ms → 100ms

**Effort estimé**: 2 semaines (1 dev + 1 ops)

---

## 5. Plan d'Implémentation par Phases

### Phase 1: Quick Wins (1-2 mois)

**Objectif**: Améliorer capacités existantes sans changement architectural majeur.

#### Sprint 1 (2 semaines)
- ✅ **Analytics Agent** (calculs YoY, MoM, trends)
  - Livrable: Module Python standalone + tests
  - Intégration: Response Generator appelle Analytics Agent
  - Impact: Support comparaisons temporelles

- ✅ **User Profile Évolutif**
  - Livrable: Nouveau modèle DB + migration
  - Tracking: Patterns queries, préférences catégories
  - Impact: Personnalisation proactive

#### Sprint 2 (2 semaines)
- ✅ **Pre-Computed Metrics** (batch nightly)
  - Livrable: Cron job + tables métriques
  - Métriques: Totaux par catégorie/marchand, YoY, MoM
  - Impact: Réponses instantanées métriques complexes

- ✅ **Query Optimizer**
  - Livrable: Module optimisation queries ES
  - Optimisations: Réordonnancement clauses, field selection
  - Impact: Réduction latence P95: 500ms → 300ms

#### Sprint 3 (2 semaines)
- ✅ **Visualisations Basiques** (line chart, bar chart, KPI cards)
  - Livrable: Composants React + API backend
  - Intégration: Response Generator génère spec viz
  - Impact: UX améliorée, compréhension données

**Résultats Phase 1**:
- Comparaisons temporelles fonctionnelles
- Réponses plus rapides (<300ms P95)
- Visualisations graphiques basiques
- Personnalisation début

### Phase 2: Analyses Avancées (2-3 mois)

**Objectif**: Support questions multi-étapes et analyses complexes.

#### Sprint 4-5 (4 semaines)
- ✅ **Reasoning Agent** (décomposition questions complexes)
  - Livrable: Agent LLM avec Chain-of-Thought
  - Framework: LangGraph ou AutoGen
  - Capabilities: Plan multi-étapes, exécution séquentielle/parallèle
  - Impact: Questions "Compare mes dépenses ce mois vs mois dernier"

#### Sprint 6 (2 semaines)
- ✅ **Dynamic Query Builder** (au-delà templates)
  - Livrable: Builder queries ES dynamiques
  - Capabilities: Agrégations imbriquées, pipelines
  - Impact: Queries multi-dimensionnelles (catégories × mois)

#### Sprint 7-8 (4 semaines)
- ✅ **Recommendation Engine** (insights actionnables)
  - Livrable: Moteur de règles + base recommandations
  - Rules: Optimisation abonnements, alertes budget, doublons
  - Intégration UI: Boutons CTA dans réponses
  - Impact: Recommandations actionnables

**Résultats Phase 2**:
- Questions complexes supportées
- Analyses multi-dimensionnelles (pivot tables)
- Recommandations actionnables avec CTA
- Raisonnement transparent (explicabilité)

### Phase 3: Scalabilité et ML (3-4 mois)

**Objectif**: Scaler à millions d'utilisateurs + capacités prédictives.

#### Sprint 9-10 (4 semaines)
- ✅ **Async Processing** (queries lentes >30s)
  - Livrable: Workers Celery/RQ + job management
  - API: Polling endpoint + webhooks
  - Impact: Pas de timeout, analyses complexes illimitées

- ✅ **Distributed Caching** (Redis Cluster)
  - Livrable: Cluster Redis 3 masters + 3 replicas
  - Migration: Cache mémoire → Redis
  - Impact: Scaling horizontal, cache hit rate 70%

#### Sprint 11-12 (4 semaines)
- ✅ **Predictive ML Models** (forecasting dépenses)
  - Livrable: Modèles scikit-learn (ARIMA, Prophet)
  - Training: Historique utilisateur 12 mois
  - Endpoints: `/predict/expenses/next_month`
  - Impact: "Prédis mes dépenses mois prochain"

#### Sprint 13-14 (4 semaines)
- ✅ **Anomaly Detection ML** (transactions suspectes)
  - Livrable: Modèle Isolation Forest + dashboard
  - Real-time: Scoring transactions entrantes
  - Alerting: Notifications push si anomalie
  - Impact: Détection fraude proactive

**Résultats Phase 3**:
- Scaling illimité (millions users)
- Forecasting dépenses/revenus
- Détection anomalies temps réel
- Cache distribué haute perf

### Phase 4: Advanced AI et Autonomous Agents (4-6 mois)

**Objectif**: IA autonome proactive + agents spécialisés.

#### Sprint 15-16 (4 semaines)
- ✅ **Autonomous Financial Advisor Agent**
  - Livrable: Agent autonome avec boucle décision
  - Capabilities: Analyses proactives, rapports automatiques
  - Triggers: Changements patterns, seuils dépassés
  - Impact: Conseils proactifs sans sollicitation utilisateur

#### Sprint 17-18 (4 semaines)
- ✅ **Multi-Agent Collaboration** (AutoGen framework)
  - Livrable: Orchestration multi-agents spécialisés
  - Agents: Analyst, Advisor, Executor, Reviewer
  - Workflow: Collaboration asynchrone + consensus
  - Impact: Analyses ultra-complexes nécessitant expertise multiple

#### Sprint 19-20 (4 semaines)
- ✅ **Reinforcement Learning from Feedback** (RLHF)
  - Livrable: Pipeline RLHF pour amélioration continue
  - Feedback: Thumbs up/down, acceptation recommandations
  - Training: Fine-tuning modèles sur préférences utilisateur
  - Impact: Réponses personnalisées optimales par utilisateur

**Résultats Phase 4**:
- IA proactive (rapports automatiques)
- Multi-agents collaboratifs
- Apprentissage continu (RLHF)
- Conseils financiers autonomes

---

## Récapitulatif Bénéfices par Phase

| Phase | Durée | Effort | Bénéfices Clés |
|-------|-------|--------|----------------|
| **Phase 1** (Quick Wins) | 1-2 mois | 2 devs | ✅ Comparaisons YoY/MoM<br>✅ Visualisations graphiques<br>✅ Performance 2x meilleure |
| **Phase 2** (Analyses Avancées) | 2-3 mois | 2-3 devs | ✅ Questions multi-étapes<br>✅ Recommandations actionnables<br>✅ Queries multi-dimensionnelles |
| **Phase 3** (Scalabilité + ML) | 3-4 mois | 2-3 devs + 1 ML eng | ✅ Scaling millions users<br>✅ Forecasting dépenses<br>✅ Détection anomalies ML |
| **Phase 4** (Advanced AI) | 4-6 mois | 3-4 devs + 1 ML eng | ✅ IA proactive autonome<br>✅ Multi-agents collaboratifs<br>✅ Apprentissage continu (RLHF) |

---

## Métriques de Succès

### Performance
- **Latence P95**: 500ms → 150ms (Phase 1)
- **Cache Hit Rate**: 30% → 70% (Phase 3)
- **Queries complexes supportées**: 0 → 100% (Phase 2)

### Capacités Fonctionnelles
- **Questions multi-étapes**: 0% → 80% (Phase 2)
- **Recommandations actionnables**: 0 → 5+ types (Phase 2)
- **Visualisations graphiques**: 0 → 7 types (Phase 1-2)

### Adoption Utilisateur
- **Taux adoption recommandations**: Target 20% (Phase 2)
- **Questions complexes /jour**: Target 15% traffic (Phase 2)
- **Satisfaction NPS**: Target +15 points (Phase 1-2)

### Scalabilité
- **Users supportés**: 10K → 1M (Phase 3)
- **Throughput**: 100 req/s → 10K req/s (Phase 3)
- **Coût par requête**: Réduction 40% (Phase 3 - caching)

---

## Conclusion

L'architecture Harena v2.0 actuelle est **solide et performante** pour des analyses financières **simples et directes**. Cependant, pour supporter des **analyses complexes**, des **conseils financiers proactifs** et des **économies personnalisées**, les améliorations proposées sont nécessaires :

### Priorités Immédiates (Phase 1-2)
1. **Analytics Agent** → Comparaisons temporelles (YoY, MoM)
2. **Reasoning Agent** → Questions multi-étapes
3. **Recommendation Engine** → Insights actionnables
4. **Visualisations Graphiques** → UX moderne

### Investissements Long-Terme (Phase 3-4)
1. **Distributed Caching** → Scaling horizontal
2. **Predictive ML** → Forecasting dépenses
3. **Autonomous Agents** → IA proactive
4. **RLHF** → Apprentissage continu

Avec ces améliorations, Harena deviendra un **véritable conseiller financier intelligent** capable d'analyses sophistiquées, de recommandations actionnables et de conseils proactifs pour aider les utilisateurs à **mieux gérer leurs finances et réaliser des économies**.
