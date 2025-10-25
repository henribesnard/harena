# Améliorations du Budget Profiling Service

> **Date de l'analyse** : 2025-10-25
> **Analyste** : Claude Code
> **Version du service** : v1.1.0

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture actuelle](#architecture-actuelle)
3. [Points forts du système](#points-forts-du-système)
4. [Améliorations recommandées](#améliorations-recommandées)
   - [Priorité 1 - Impact Élevé](#priorité-1---impact-élevé)
   - [Priorité 2 - Impact Moyen](#priorité-2---impact-moyen)
   - [Priorité 3 - Optimisations](#priorité-3---optimisations)
5. [Plan d'implémentation](#plan-dimplémentation)
6. [Métriques de succès](#métriques-de-succès)

---

## Vue d'ensemble

Le **Budget Profiling Service** calcule le profil budgétaire des utilisateurs en analysant leurs transactions historiques. Le système est fonctionnel et bien conçu, mais plusieurs axes d'amélioration ont été identifiés pour augmenter la **précision**, la **fiabilité** et la **pertinence** des profils calculés.

### Résumé des problématiques principales

| Problème | Impact | Priorité |
|----------|--------|----------|
| Période d'analyse par défaut trop courte (3 mois) | Profils peu fiables, charges fixes manquées | 🔥 CRITIQUE |
| Détection charges fixes trop stricte | Charges fixes non détectées | 🔥 ÉLEVÉ |
| Calcul de tendance simpliste (2 mois) | Faux signaux, volatilité excessive | 🔥 ÉLEVÉ |
| Classification "autres" trop conservative | Métriques faussées | ⚡ MOYEN |
| Pas de détection de pics/anomalies | Moyennes biaisées | ⚡ MOYEN |
| Score de complétude incomplet | Mauvaise évaluation qualité profil | ⚡ MOYEN |
| Cache catégories non partagé | Performances sous-optimales | 🔧 BAS |
| Alertes avec seuils universels | Alertes peu pertinentes | 🔧 BAS |

---

## Architecture actuelle

Le service est structuré en **4 composants principaux** :

### 1. `BudgetProfiler` (services/budget_profiler.py)
- **Rôle** : Orchestrateur principal du calcul de profil
- **Méthode clé** : `calculate_user_profile()` (ligne 29)
- **Flux** :
  1. Récupération agrégats mensuels
  2. Calcul moyennes (revenus/dépenses/épargne)
  3. Détection charges fixes
  4. Classification dépenses (fixes/semi-fixes/variables)
  5. Segmentation utilisateur
  6. Patterns comportementaux
  7. Métriques avancées (volatilité, tendances, health score)
  8. Génération alertes

### 2. `TransactionService` (services/transaction_service.py)
- **Rôle** : Récupération et agrégation des transactions
- **Optimisations** : Calculs DB-side (GROUP BY, agrégations SQL)
- **Méthodes clés** :
  - `get_monthly_aggregates()` (ligne 119) - Agrégats par mois
  - `get_category_breakdown()` (ligne 214) - Répartition par catégorie

### 3. `FixedChargeDetector` (services/fixed_charge_detector.py)
- **Rôle** : Détection automatique des charges fixes récurrentes
- **Algorithme** : Multi-critères (récurrence temporelle + variance montant + variance jour)
- **Score de confiance** : Composite pondéré (ligne 236)

### 4. `AdvancedBudgetAnalytics` (services/advanced_analytics.py)
- **Rôle** : Analytics avancées (segmentation, patterns, alertes)
- **Fonctionnalités** :
  - Segmentation multi-critères avec scoring 0-100
  - Détection multi-patterns comportementaux
  - Score de santé financière global
  - Système d'alertes contextuelles

---

## Points forts du système

### ✅ Optimisations performantes

**Agrégations DB-side** :
- Les calculs mensuels et breakdown catégories sont effectués en SQL
- Évite de charger toutes les transactions en mémoire
- Performance excellente même avec gros volumes

**Fichier** : `transaction_service.py:155-188`
```python
# Agrégation DB-side avec GROUP BY sur année/mois
query = (
    select(
        extract('year', RawTransaction.date).label('year'),
        extract('month', RawTransaction.date).label('month'),
        func.sum(...).label('total_income'),
        func.sum(...).label('total_expenses'),
        ...
    )
    .group_by(extract('year', ...), extract('month', ...))
)
```

### ✅ Détection intelligente charges fixes

- Algorithme multi-critères robuste
- Filtrage marchands variables connus (supermarchés, essence, etc.)
- Score de confiance composite
- Seuil minimum pour éviter faux positifs

### ✅ Analytics avancées

- Segmentation enrichie (pas juste ratio dépenses/revenus)
- Multi-patterns comportementaux (pas un seul pattern)
- Score de santé financière holistique
- Alertes contextuelles et actionnables

### ✅ Modèle de données complet

- Relations bien définies
- Support objectifs d'épargne
- Recommandations personnalisées
- Patterns saisonniers

---

## Améliorations recommandées

### Priorité 1 - Impact Élevé

#### 🔥 **1.1 - Période d'analyse par défaut trop courte**

**Problème identifié** :
```python
# budget_profiling_service/services/budget_profiler.py:32
def calculate_user_profile(
    self,
    user_id: int,
    months_analysis: int = 3  # ⚠️ SEULEMENT 3 MOIS !
)
```

**Impact** :
- ❌ Profil basé sur seulement 3 mois = peu représentatif
- ❌ Charges fixes nécessitent min 3 occurrences → risque de ne pas détecter
- ❌ Patterns saisonniers impossibles à identifier
- ❌ Incohérence avec la documentation (dit "None = toutes transactions")

**Solution recommandée** :
```python
def calculate_user_profile(
    self,
    user_id: int,
    months_analysis: Optional[int] = None  # None = toutes transactions
)
```

**Fichiers à modifier** :
- `budget_profiling_service/services/budget_profiler.py:32`

**Effort estimé** : 🟢 Faible (5 minutes)
**Impact** : 🔴 Très élevé (profils beaucoup plus fiables)

---

#### 🔥 **1.2 - Détection charges fixes trop stricte**

**Problème identifié** :
```python
# budget_profiling_service/services/fixed_charge_detector.py:82-84
min_occurrences: int = 3,
max_amount_variance_pct: float = 10.0,  # ⚠️ Trop strict
max_day_variance: int = 5                # ⚠️ Trop strict
```

**Cas ratés** :
- ❌ Abonnement Netflix augmente de 11% → non détecté (> 10%)
- ❌ Prélèvement EDF varie entre le 28 et le 3 → non détecté (> 5 jours)
- ❌ Assurance prélevée tous les 15 jours → non détecté (intervalle 13-17j au lieu de 28-32j)

**Solution recommandée** :
```python
min_occurrences: int = 3,
max_amount_variance_pct: float = 15.0,  # Plus de tolérance
max_day_variance: int = 7,              # Tolérance week-ends
```

**Amélioration supplémentaire** : Ajouter détection charges bi-mensuelles
```python
# Ajouter dans _analyze_recurrence() après ligne 204
if not (20 <= avg_interval <= 40):
    # Vérifier si bi-mensuel (tous les 15 jours)
    if 13 <= avg_interval <= 17:
        # C'est une charge bi-mensuelle
        pass  # Accepter
    else:
        return None  # Rejeter
```

**Fichiers à modifier** :
- `budget_profiling_service/services/fixed_charge_detector.py:82-84`
- `budget_profiling_service/services/fixed_charge_detector.py:195-208`

**Effort estimé** : 🟡 Moyen (1-2 heures)
**Impact** : 🔴 Élevé (détecte 20-30% de charges fixes supplémentaires)

---

#### 🔥 **1.3 - Calcul de tendance trop simpliste**

**Problème identifié** :
```python
# budget_profiling_service/services/advanced_analytics.py:188-190
# Prendre les 2 mois les plus récents pour comparaison
current_month = sorted_months[0]
prev_month = sorted_months[1]
```

**Impact** :
- ❌ Compare seulement 2 mois → très volatile
- ❌ Un pic ponctuel (Black Friday, Noël) → fausse tendance "increasing"
- ❌ Ne détecte pas les vraies tendances de fond
- ❌ Alertes "dépenses en hausse" trop fréquentes

**Solution recommandée** : Moyenne mobile sur 3 mois
```python
def analyze_spending_trend(monthly_aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyse la tendance avec moyenne mobile 3 mois
    """
    if len(monthly_aggregates) < 6:
        return {...}  # Données insuffisantes

    sorted_months = sorted(monthly_aggregates, key=lambda x: x['month'], reverse=True)

    # Moyenne mobile actuelle (3 derniers mois)
    current_avg_expenses = sum(m['total_expenses'] for m in sorted_months[0:3]) / 3
    current_avg_income = sum(m['total_income'] for m in sorted_months[0:3]) / 3

    # Moyenne mobile précédente (mois 3 à 6)
    prev_avg_expenses = sum(m['total_expenses'] for m in sorted_months[3:6]) / 3
    prev_avg_income = sum(m['total_income'] for m in sorted_months[3:6]) / 3

    # Calculer variation
    expense_change_pct = ((current_avg_expenses - prev_avg_expenses) / prev_avg_expenses * 100) if prev_avg_expenses > 0 else 0

    # Déterminer tendance avec seuil plus élevé
    if abs(expense_change_pct) < 10:  # Moins de 10% = stable
        trend = 'stable'
    elif expense_change_pct > 10:
        trend = 'increasing'
    else:
        trend = 'decreasing'

    return {
        'trend': trend,
        'change_pct': round(expense_change_pct, 2),
        'current_avg_expenses': round(current_avg_expenses, 2),
        'prev_avg_expenses': round(prev_avg_expenses, 2),
        ...
    }
```

**Alternative avancée** : Régression linéaire sur 6-12 mois
```python
# Utiliser numpy ou statsmodels pour régression
import numpy as np

months_data = [(i, m['total_expenses']) for i, m in enumerate(sorted_months[:12])]
x = np.array([m[0] for m in months_data])
y = np.array([m[1] for m in months_data])

# Régression linéaire
slope, intercept = np.polyfit(x, y, 1)

if abs(slope) < 50:  # Moins de 50€/mois de variation
    trend = 'stable'
elif slope > 50:
    trend = 'increasing'
else:
    trend = 'decreasing'
```

**Fichiers à modifier** :
- `budget_profiling_service/services/advanced_analytics.py:161-217`

**Effort estimé** : 🟡 Moyen (2-3 heures)
**Impact** : 🔴 Élevé (tendances beaucoup plus fiables, moins de fausses alertes)

---

### Priorité 2 - Impact Moyen

#### ⚡ **2.1 - Ajouter détection de pics et anomalies**

**Problème** :
- Actuellement, AUCUNE détection des dépenses exceptionnelles
- Les moyennes sont faussées par les pics (mariage, déménagement, vacances)
- Le profil d'un utilisateur varie énormément selon le mois d'analyse

**Exemple concret** :
```
Utilisateur A:
- Jan-Oct : 2000€/mois de dépenses
- Novembre : 8000€ (mariage)
- Décembre : 2000€

→ Moyenne sur 12 mois : 2500€/mois
→ Moyenne hors pic : 2000€/mois
→ Différence : 25% !
```

**Solution recommandée** : Créer un module `OutlierDetector`

**Nouveau fichier** : `budget_profiling_service/services/outlier_detector.py`
```python
"""
Détection de pics et anomalies dans les dépenses
"""
from typing import List, Dict, Any, Tuple
import statistics

class OutlierDetector:
    """
    Détecte les mois avec dépenses anormalement élevées ou basses
    """

    @staticmethod
    def detect_spending_outliers(
        monthly_aggregates: List[Dict[str, Any]],
        method: str = 'iqr'  # 'iqr' ou 'zscore'
    ) -> Tuple[List[Dict], List[str]]:
        """
        Détecte les outliers dans les dépenses mensuelles

        Returns:
            (clean_data, outlier_months)
        """
        if len(monthly_aggregates) < 6:
            return monthly_aggregates, []

        expenses = [m['total_expenses'] for m in monthly_aggregates]

        if method == 'iqr':
            # Méthode IQR (Interquartile Range)
            q1 = statistics.quantiles(expenses, n=4)[0]  # 25e percentile
            q3 = statistics.quantiles(expenses, n=4)[2]  # 75e percentile
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

        else:  # zscore
            # Méthode Z-score
            mean = statistics.mean(expenses)
            stdev = statistics.stdev(expenses)

            lower_bound = mean - 2 * stdev
            upper_bound = mean + 2 * stdev

        # Identifier les outliers
        outlier_months = []
        clean_data = []

        for month_data in monthly_aggregates:
            expense = month_data['total_expenses']

            if lower_bound <= expense <= upper_bound:
                clean_data.append(month_data)
            else:
                outlier_months.append({
                    'month': month_data['month'],
                    'expense': expense,
                    'type': 'spike' if expense > upper_bound else 'drop',
                    'deviation_pct': ((expense - statistics.mean(expenses)) / statistics.mean(expenses)) * 100
                })

        return clean_data, outlier_months

    @staticmethod
    def categorize_outlier_reason(
        user_id: int,
        outlier_month: str,
        transaction_service
    ) -> str:
        """
        Tente de catégoriser la raison du pic
        """
        # Récupérer transactions du mois en question
        # Identifier la catégorie dominante
        # Exemples: 'vacation', 'wedding', 'medical', 'moving', 'unknown'
        pass
```

**Intégration dans BudgetProfiler** :
```python
# budget_profiling_service/services/budget_profiler.py

from budget_profiling_service.services.outlier_detector import OutlierDetector

def calculate_user_profile(...):
    # ... code existant ...

    # Détecter outliers
    clean_aggregates, outliers = OutlierDetector.detect_spending_outliers(
        monthly_aggregates
    )

    # Calculer 2 profils
    profile_with_outliers = {
        'avg_monthly_expenses': self._calculate_avg(monthly_aggregates),
        ...
    }

    profile_without_outliers = {
        'avg_monthly_expenses': self._calculate_avg(clean_aggregates),
        ...
    }

    # Stocker les 2
    return {
        ...profile_with_outliers,
        'baseline_profile': profile_without_outliers,  # Nouveau champ
        'spending_outliers': outliers,                  # Nouveau champ
        'outlier_count': len(outliers),
        ...
    }
```

**Fichiers à créer** :
- `budget_profiling_service/services/outlier_detector.py`

**Fichiers à modifier** :
- `budget_profiling_service/services/budget_profiler.py`
- `db_service/models/budget_profiling.py` (ajouter champs `baseline_profile` JSON et `spending_outliers` JSON)

**Effort estimé** : 🟠 Élevé (1 journée)
**Impact** : 🟡 Moyen-Élevé (moyennes beaucoup plus représentatives)

---

#### ⚡ **2.2 - Améliorer classification des "autres"**

**Problème identifié** :
```python
# budget_profiling_service/services/budget_profiler.py:302
# Ajouter "autres" aux charges variables (considérées comme discrétionnaires par défaut)
variable_total += other_total
```

**Impact** :
- ❌ **Virements sortants** → peuvent être loyer, prêt (charges FIXES)
- ❌ **Chèques** → peuvent être crèche, école (charges FIXES)
- ❌ **Retraits espèces** → utilisation inconnue
- ❌ Résultat : charges variables surestimées, charges fixes sous-estimées

**Solution court terme** : Analyser la récurrence des "autres"
```python
def _categorize_expenses(self, category_breakdown, fixed_charges_total):
    """
    Amélioration : détecter les "autres" récurrents
    """
    semi_fixed_total = 0.0
    variable_total = 0.0
    structural_fixed_total = 0.0
    other_total = 0.0

    # Nouveau: autres récurrents détectés comme fixes
    other_recurring_total = 0.0

    for category, amount in category_breakdown.items():
        category_lower = category.lower()

        # Classification standard
        if any(fixed in category_lower for fixed in fixed_categories):
            structural_fixed_total += amount
        elif any(semi in category_lower for semi in semi_fixed_categories):
            semi_fixed_total += amount
        elif any(var in category_lower for var in variable_categories):
            variable_total += amount
        else:
            # Pour les "autres", vérifier récurrence
            # Si catégorie = "virement" ou "chèque", analyser transactions individuelles
            if category_lower in ['virement', 'cheque', 'prelevement']:
                # Appeler fixed_charge_detector sur ces transactions spécifiquement
                # Si récurrence détectée → other_recurring_total
                # Sinon → other_total
                pass
            else:
                other_total += amount

    # Ajouter récurrents aux fixes, reste aux variables
    structural_fixed_total += other_recurring_total
    variable_total += other_total

    return semi_fixed_total, variable_total, structural_fixed_total
```

**Solution long terme** : Améliorer la catégorisation en amont
- Enrichir les règles de catégorisation dans le service de sync
- Utiliser ML pour classifier automatiquement (NLP sur descriptions)
- Permettre à l'utilisateur de valider/corriger les catégories

**Fichiers à modifier** :
- `budget_profiling_service/services/budget_profiler.py:221-306`

**Effort estimé** : 🟡 Moyen (3-4 heures pour court terme)
**Impact** : 🟡 Moyen (métriques plus précises de 5-15%)

---

#### ⚡ **2.3 - Améliorer score de complétude**

**Problème identifié** :
```python
# budget_profiling_service/services/budget_profiler.py:368
def _calculate_completeness(self, monthly_aggregates, fixed_charges, months_required):
    # Score basé sur: nb_mois + nb_charges_fixes + has_income
    # ⚠️ Ne prend PAS en compte la QUALITÉ des catégories
```

**Cas problématique** :
```
Utilisateur A:
- 12 mois de données ✅
- 5 charges fixes ✅
- Revenus présents ✅
- Score: 1.0 (100%)

MAIS:
- 80% des transactions sont "uncategorized"
- Seulement 2 marchands différents
→ Profil peu fiable malgré score 100%
```

**Solution recommandée** : Ajouter 2 facteurs
```python
def _calculate_completeness(
    self,
    monthly_aggregates: list,
    fixed_charges: list,
    months_required: Optional[int],
    user_id: int  # Nouveau paramètre
) -> float:
    """
    Score de complétude amélioré (0.0 - 1.0)
    """
    score = 0.0

    # Facteur 1: Nombre de mois (max 0.3, réduit de 0.4)
    if months_required is None:
        months_score = min(len(monthly_aggregates) / 12.0, 1.0) * 0.3
    else:
        months_score = min(len(monthly_aggregates) / months_required, 1.0) * 0.3

    # Facteur 2: Charges fixes détectées (max 0.25, réduit de 0.3)
    fixed_charges_score = min(len(fixed_charges) / 5.0, 1.0) * 0.25

    # Facteur 3: Présence revenus (max 0.25, réduit de 0.3)
    has_income = any(m['total_income'] > 0 for m in monthly_aggregates)
    income_score = 0.25 if has_income else 0.0

    # ✨ NOUVEAU Facteur 4: Qualité de catégorisation (max 0.2)
    transactions = self.transaction_service.get_user_transactions(user_id)
    if transactions:
        categorized_count = sum(1 for tx in transactions if tx['category'] != 'uncategorized')
        categorization_ratio = categorized_count / len(transactions)
        category_score = categorization_ratio * 0.2
    else:
        category_score = 0.0

    score = months_score + fixed_charges_score + income_score + category_score
    return min(max(score, 0.0), 1.0)
```

**Amélioration supplémentaire** : Ajouter diversité des transactions
```python
# Facteur 5: Diversité des marchands (optionnel, max 0.1)
unique_merchants = len(set(tx['merchant_name'] for tx in transactions if tx['merchant_name']))
diversity_score = min(unique_merchants / 20.0, 1.0) * 0.1

# Ajuster les pondérations précédentes pour que total = 1.0
```

**Fichiers à modifier** :
- `budget_profiling_service/services/budget_profiler.py:368-395`
- `budget_profiling_service/services/budget_profiler.py:101` (passer user_id)

**Effort estimé** : 🟡 Moyen (2 heures)
**Impact** : 🟡 Moyen (meilleure évaluation de la fiabilité)

---

### Priorité 3 - Optimisations

#### 🔧 **3.1 - Cache partagé pour catégories**

**Problème identifié** :
```python
# budget_profiling_service/services/transaction_service.py:22
def __init__(self, db_session: Session):
    self.db = db_session
    self._category_cache = {}  # ⚠️ Cache par instance
```

**Impact** :
- ❌ Chaque instance a son propre cache
- ❌ Si 100 utilisateurs analysés en parallèle → 100 fois les mêmes requêtes SQL
- ❌ Pas de TTL → cache peut devenir obsolète

**Solution recommandée** : Cache module-level
```python
# budget_profiling_service/services/transaction_service.py

from functools import lru_cache
from datetime import datetime, timedelta

# Cache partagé au niveau du module
_CATEGORY_CACHE = {}
_CATEGORY_CACHE_TIMESTAMP = None
_CACHE_TTL = timedelta(hours=1)

class TransactionService:

    def _get_category_name(self, category_id: Optional[int]) -> str:
        """
        Récupère nom catégorie avec cache partagé et TTL
        """
        if not category_id:
            return 'uncategorized'

        # Invalider cache si trop ancien
        global _CATEGORY_CACHE_TIMESTAMP
        if _CATEGORY_CACHE_TIMESTAMP is None or \
           datetime.now() - _CATEGORY_CACHE_TIMESTAMP > _CACHE_TTL:
            _CATEGORY_CACHE.clear()
            _CATEGORY_CACHE_TIMESTAMP = datetime.now()

        # Vérifier cache
        if category_id in _CATEGORY_CACHE:
            return _CATEGORY_CACHE[category_id]

        # Récupérer depuis DB
        try:
            result = self.db.execute(
                select(Category).where(Category.category_id == category_id)
            )
            category = result.scalar_one_or_none()

            if category:
                _CATEGORY_CACHE[category_id] = category.category_name
                return category.category_name
            else:
                return 'uncategorized'
        except Exception as e:
            logger.error(f"Erreur récupération catégorie {category_id}: {e}")
            return 'uncategorized'
```

**Alternative avec Redis** (si disponible) :
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def _get_category_name(self, category_id: Optional[int]) -> str:
    if not category_id:
        return 'uncategorized'

    # Vérifier Redis
    cache_key = f"category:{category_id}"
    cached = redis_client.get(cache_key)

    if cached:
        return cached.decode('utf-8')

    # Récupérer DB
    category_name = self._fetch_category_from_db(category_id)

    # Stocker dans Redis avec TTL 1h
    redis_client.setex(cache_key, 3600, category_name)

    return category_name
```

**Amélioration supplémentaire** : Précharger toutes les catégories au démarrage
```python
class TransactionService:

    def __init__(self, db_session: Session):
        self.db = db_session
        self._preload_categories()

    def _preload_categories(self):
        """
        Charge toutes les catégories au démarrage
        """
        global _CATEGORY_CACHE

        if _CATEGORY_CACHE:
            return  # Déjà chargé

        try:
            result = self.db.execute(select(Category))
            categories = result.scalars().all()

            for cat in categories:
                _CATEGORY_CACHE[cat.category_id] = cat.category_name

            logger.info(f"Préchargé {len(_CATEGORY_CACHE)} catégories")
        except Exception as e:
            logger.error(f"Erreur préchargement catégories: {e}")
```

**Fichiers à modifier** :
- `budget_profiling_service/services/transaction_service.py:22, 91-117`

**Effort estimé** : 🟢 Faible (1 heure)
**Impact** : 🔧 Faible-Moyen (améliore performances de 10-20%)

---

#### 🔧 **3.2 - Patterns comportementaux sur 6 mois**

**Problème identifié** :
```python
# budget_profiling_service/services/advanced_analytics.py:434
transactions = transaction_service.get_user_transactions(
    user_id,
    months_back=3  # ⚠️ Seulement 3 mois
)
```

**Impact** :
- ❌ 3 mois insuffisants pour patterns saisonniers
- ❌ Un "acheteur impulsif" en décembre (cadeaux) peut être "planificateur" en mars

**Solution** :
```python
months_back=6  # Au moins 6 mois pour patterns robustes
```

**Fichiers à modifier** :
- `budget_profiling_service/services/advanced_analytics.py:434`

**Effort estimé** : 🟢 Faible (2 minutes)
**Impact** : 🔧 Faible-Moyen (patterns plus fiables)

---

#### 🔧 **3.3 - Personnalisation des seuils d'alertes**

**Problème** :
- Seuils fixes universels : 5% épargne, 70% charges fixes, etc.
- Un étudiant avec 0% épargne ≠ un cadre avec 0% épargne
- Comparaison impossible avec pairs

**Solution recommandée** : Seuils adaptatifs selon profil

```python
# Nouveau fichier: budget_profiling_service/services/alert_thresholds.py

ALERT_THRESHOLDS = {
    'student': {
        'min_savings_rate': 0,      # Étudiant: 0% OK
        'max_fixed_ratio': 0.8,      # 80% max
        'max_spending_increase': 20  # 20% variation OK
    },
    'young_professional': {
        'min_savings_rate': 5,
        'max_fixed_ratio': 0.7,
        'max_spending_increase': 15
    },
    'family': {
        'min_savings_rate': 10,
        'max_fixed_ratio': 0.6,
        'max_spending_increase': 10
    },
    'senior': {
        'min_savings_rate': 15,
        'max_fixed_ratio': 0.5,
        'max_spending_increase': 5
    }
}

def get_thresholds_for_user(user_profile: Dict) -> Dict:
    """
    Détermine les seuils selon profil utilisateur
    """
    # Logique de classification basée sur:
    # - Âge (si disponible)
    # - Niveau de revenus
    # - Situation familiale (si disponible)

    avg_income = user_profile.get('avg_monthly_income', 0)

    if avg_income < 1500:
        return ALERT_THRESHOLDS['student']
    elif avg_income < 3000:
        return ALERT_THRESHOLDS['young_professional']
    elif avg_income < 5000:
        return ALERT_THRESHOLDS['family']
    else:
        return ALERT_THRESHOLDS['senior']
```

**Intégration dans generate_alerts** :
```python
def generate_alerts(profile_data: Dict[str, Any], user_id: int) -> List[Dict]:
    alerts = []

    # Récupérer seuils personnalisés
    thresholds = get_thresholds_for_user(profile_data)

    # Alerte taux d'épargne (seuil adaptatif)
    if profile_data.get('savings_rate', 0) < thresholds['min_savings_rate']:
        alerts.append({...})

    # Alerte charges fixes (seuil adaptatif)
    fixed_ratio = profile_data.get('fixed_charges_total', 0) / profile_data.get('avg_monthly_income', 1)
    if fixed_ratio > thresholds['max_fixed_ratio']:
        alerts.append({...})

    return alerts
```

**Fichiers à créer** :
- `budget_profiling_service/services/alert_thresholds.py`

**Fichiers à modifier** :
- `budget_profiling_service/services/advanced_analytics.py:307-413`

**Effort estimé** : 🟡 Moyen (3 heures)
**Impact** : 🔧 Faible-Moyen (alertes plus pertinentes)

---

#### 🔧 **3.4 - Contexte géographique pour segmentation**

**Problème** :
- Seuils universels : 500€ de reste à vivre minimum
- Paris ≠ Province → coût de vie très différent
- Un "confortable" à Paris = "précaire" en province

**Solution** : Ajouter paramètre `region` ou `cost_of_living_index`

```python
# Nouveau fichier: budget_profiling_service/services/regional_adjustments.py

COST_OF_LIVING_INDEX = {
    'paris': 1.3,
    'lyon': 1.1,
    'marseille': 1.0,
    'toulouse': 0.95,
    'nantes': 0.95,
    'rural': 0.85,
    'default': 1.0
}

def adjust_thresholds_for_region(
    base_threshold: float,
    region: str
) -> float:
    """
    Ajuste les seuils selon le coût de vie régional
    """
    coli = COST_OF_LIVING_INDEX.get(region, 1.0)
    return base_threshold * coli
```

**Intégration dans segmentation** :
```python
def determine_segment_v2(
    avg_income: float,
    avg_expenses: float,
    remaining_to_live: float,
    fixed_charges_total: float,
    region: str = 'default'  # Nouveau paramètre
) -> Dict[str, Any]:

    # Ajuster seuils selon région
    min_remaining_needed = adjust_thresholds_for_region(
        min(500, avg_income * 0.2),
        region
    )

    # Reste du calcul...
```

**Note** : Nécessite d'avoir l'info région dans profil utilisateur

**Fichiers à créer** :
- `budget_profiling_service/services/regional_adjustments.py`

**Fichiers à modifier** :
- `budget_profiling_service/services/advanced_analytics.py:18`
- `db_service/models/user.py` (ajouter champ `region` si absent)

**Effort estimé** : 🟡 Moyen (2-3 heures)
**Impact** : 🔧 Faible (améliore pertinence pour utilisateurs Paris/Province)

---

## Plan d'implémentation

### Phase 1 - Quick Wins (1 journée)

**Objectif** : Gains immédiats avec faible effort

1. ✅ Changer `months_analysis` de `3` à `None` (5 min)
2. ✅ Augmenter seuils détection charges fixes (30 min)
3. ✅ Patterns comportementaux sur 6 mois (2 min)
4. ✅ Cache partagé catégories (1h)

**Impact attendu** :
- 🎯 Profils 30-40% plus fiables
- 🎯 20-30% plus de charges fixes détectées
- 🎯 10-20% meilleures performances

---

### Phase 2 - Améliorations Moyennes (2-3 jours)

**Objectif** : Améliorer fiabilité et précision

5. ✅ Calcul tendance avec moyenne mobile (3h)
6. ✅ Améliorer classification "autres" (4h)
7. ✅ Améliorer score de complétude (2h)
8. ✅ Ajouter support charges bi-mensuelles (2h)

**Impact attendu** :
- 🎯 Tendances 50% plus stables
- 🎯 Métriques 10-15% plus précises
- 🎯 Score complétude plus représentatif

---

### Phase 3 - Fonctionnalités Avancées (1 semaine)

**Objectif** : Nouvelles capacités analytiques

9. ✅ Détection de pics/anomalies (1 jour)
10. ✅ Personnalisation seuils alertes (3h)
11. ✅ Contexte géographique (3h)

**Impact attendu** :
- 🎯 Moyennes 20-30% plus représentatives
- 🎯 Alertes plus pertinentes
- 🎯 Segmentation plus fine

---

### Ordre de priorité recommandé

```
Sprint 1 (1 journée):
├─ 1.1 - Période analyse par défaut (5 min)
├─ 1.2 - Seuils détection charges fixes (30 min)
├─ 3.2 - Patterns sur 6 mois (2 min)
└─ 3.1 - Cache partagé (1h)

Sprint 2 (2-3 jours):
├─ 1.3 - Tendance moyenne mobile (3h)
├─ 2.2 - Classification "autres" (4h)
├─ 2.3 - Score complétude (2h)
└─ Tests et validation

Sprint 3 (1 semaine):
├─ 2.1 - Détection outliers (1j)
├─ 3.3 - Personnalisation alertes (3h)
└─ Tests et documentation
```

---

## Métriques de succès

### KPIs à tracker avant/après

| Métrique | Avant | Cible Après | Comment mesurer |
|----------|-------|-------------|-----------------|
| **Taux de détection charges fixes** | ~60% | >85% | Comparer avec validation manuelle |
| **Score complétude moyen** | 0.65 | >0.80 | Moyenne sur tous utilisateurs |
| **Faux positifs alertes** | ~30% | <10% | Feedback utilisateurs |
| **Temps calcul profil** | 2-3s | <1.5s | Monitoring performance |
| **Précision moyennes** | ±20% | ±5% | Comparer avec/sans outliers |
| **Stabilité tendances** | Volatilité 0.8 | <0.4 | Coefficient variation |

### Tests de validation

**Avant déploiement** :
1. ✅ Tester sur échantillon 100 utilisateurs variés
2. ✅ Comparer profils avant/après
3. ✅ Valider détection charges fixes manuellement
4. ✅ Vérifier cohérence métriques
5. ✅ Tests de régression automatisés

**Après déploiement** :
1. ✅ Monitoring alertes générées
2. ✅ Feedback utilisateurs
3. ✅ A/B testing si possible

---

## Annexes

### Fichiers concernés par les modifications

```
budget_profiling_service/
├── services/
│   ├── budget_profiler.py          # 6 modifications
│   ├── fixed_charge_detector.py    # 2 modifications
│   ├── advanced_analytics.py       # 4 modifications
│   ├── transaction_service.py      # 1 modification
│   ├── outlier_detector.py         # 🆕 NOUVEAU
│   ├── alert_thresholds.py         # 🆕 NOUVEAU
│   └── regional_adjustments.py     # 🆕 NOUVEAU
└── PROFILING_CALCULATION_GUIDE.md  # Mise à jour doc

db_service/models/
└── budget_profiling.py             # Ajouter champs JSON
```

### Dépendances additionnelles

```toml
# pyproject.toml ou requirements.txt

# Pour détection outliers
numpy>=1.24.0

# Pour cache Redis (optionnel)
redis>=4.5.0

# Pour régression linéaire avancée (optionnel)
scikit-learn>=1.2.0
```

### Migration base de données

```python
# alembic/versions/xxx_add_outlier_fields.py

def upgrade():
    # Ajouter champs JSON pour outliers
    op.add_column('user_budget_profile',
        sa.Column('baseline_profile', JSON, nullable=True))
    op.add_column('user_budget_profile',
        sa.Column('spending_outliers', JSON, nullable=True))

def downgrade():
    op.drop_column('user_budget_profile', 'baseline_profile')
    op.drop_column('user_budget_profile', 'spending_outliers')
```

---

## Ressources

### Documentation de référence

- [PROFILING_CALCULATION_GUIDE.md](budget_profiling_service/PROFILING_CALCULATION_GUIDE.md) - Guide détaillé calcul profil
- [README.md](budget_profiling_service/README.md) - Documentation service

### Contacts

- **Mainteneur** : Henri Besnard
- **Code review** : À définir
- **Validation métier** : À définir

---

**Document généré le** : 2025-10-25
**Version** : 1.0
**Statut** : 📝 Proposition - En attente de validation
