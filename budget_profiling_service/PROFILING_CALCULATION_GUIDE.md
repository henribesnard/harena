# Guide de Calcul du Profil Budgétaire

## Vue d'ensemble

Le service de **Budget Profiling** analyse l'historique des transactions d'un utilisateur pour calculer un profil budgétaire complet. Ce document explique en détail comment chaque élément du profil est calculé.

---

## Table des matières

1. [Données source](#données-source)
2. [Période d'analyse](#période-danalyse)
3. [Moyennes mensuelles](#moyennes-mensuelles)
4. [Détection des charges fixes](#détection-des-charges-fixes)
5. [Classification des dépenses](#classification-des-dépenses)
6. [Métriques calculées](#métriques-calculées)
7. [Segmentation utilisateur](#segmentation-utilisateur)
8. [Pattern comportemental](#pattern-comportemental)
9. [Complétude du profil](#complétude-du-profil)

---

## Données source

### Table utilisée
- **Table** : `raw_transactions`
- **Filtres** :
  - `user_id = {id_utilisateur}`
  - `deleted = false`
  - `date >= start_date AND date <= end_date`

### Champs exploités
- `amount` : Montant de la transaction
- `date` : Date de la transaction
- `category_id` : Référence à la table `categories`
- `merchant_name` : Nom du marchand
- `operation_type` : Type d'opération

---

## Période d'analyse

### Par défaut (sans paramètre)
```
months_analysis = None
→ Analyse TOUTES les transactions disponibles depuis 2000-01-01
```

### Avec paramètre explicite
```
months_analysis = 12
→ Analyse les 12 derniers mois uniquement
start_date = now() - (30 jours × 12)
```

### Calcul du nombre de mois réels

**Important** : On ne compte que les mois **avec au moins une transaction** :

```python
# Grouper toutes les transactions par mois
all_months = set()
for tx in transactions:
    month_key = f"{tx.date.year}-{tx.date.month:02d}"
    all_months.add(month_key)

nb_months_reels = len(all_months)
```

**Exemple** :
- Période : Janvier 2020 → Octobre 2025 (70 mois calendaires)
- Mois sans transactions : 10 mois
- **Nombre de mois réels** : 60 mois

---

## Moyennes mensuelles

### 1. Revenus moyens (`avg_monthly_income`)

**Calcul** :
```python
# Grouper par mois
monthly_aggregates = grouper_par_mois(transactions)

# Pour chaque mois, sommer les crédits
for month in monthly_aggregates:
    total_income = sum(tx.amount for tx in month if tx.amount > 0)

# Moyenne sur tous les mois
avg_monthly_income = sum(month.total_income for month in monthly_aggregates) / nb_months
```

**Exemple** :
```
Mois 1 : 2500 EUR
Mois 2 : 2800 EUR
Mois 3 : 2600 EUR
→ Moyenne : (2500 + 2800 + 2600) / 3 = 2633.33 EUR
```

### 2. Dépenses moyennes (`avg_monthly_expenses`)

**Calcul** :
```python
# Pour chaque mois, sommer les débits (valeur absolue)
for month in monthly_aggregates:
    total_expenses = sum(abs(tx.amount) for tx in month if tx.amount < 0)

# Moyenne sur tous les mois
avg_monthly_expenses = sum(month.total_expenses for month in monthly_aggregates) / nb_months
```

### 3. Épargne moyenne (`avg_monthly_savings`)

**Calcul** :
```python
avg_monthly_savings = avg_monthly_income - avg_monthly_expenses
```

**Exemple** :
```
Revenus : 2633 EUR
Dépenses : 2200 EUR
→ Épargne : 433 EUR
```

### 4. Taux d'épargne (`savings_rate`)

**Calcul** :
```python
savings_rate = (avg_monthly_savings / avg_monthly_income) × 100
```

**Exemple** :
```
Épargne : 433 EUR
Revenus : 2633 EUR
→ Taux : (433 / 2633) × 100 = 16.44%
```

**Cas particuliers** :
- Si `avg_monthly_income = 0` → `savings_rate = 0.0%`
- Le taux peut être **négatif** si dépenses > revenus

---

## Détection des charges fixes

### Définition
Les **charges fixes** sont des dépenses récurrentes à montant et date (quasi) identiques chaque mois.

### Critères de détection

**Paramètres par défaut** :
- `min_occurrences = 3` : Minimum 3 occurrences pour confirmer
- `max_amount_variance_pct = 10%` : Variance max du montant
- `max_day_variance = 5` : Variance max du jour du mois

### Algorithme

**Étape 1** : Grouper par marchand
```python
merchant_groups = defaultdict(list)
for tx in debits:
    if tx.merchant_name:
        merchant_groups[tx.merchant_name].append(tx)
```

**Étape 2** : Pour chaque marchand, analyser la récurrence
```python
# Trier chronologiquement
sorted_txs = sorted(transactions, key=lambda x: x.date)

# Calculer variance du montant
amounts = [abs(tx.amount) for tx in sorted_txs]
avg_amount = mean(amounts)
amount_variance_pct = (stdev(amounts) / avg_amount) × 100

# Vérifier variance montant
if amount_variance_pct > 10%:
    return None  # Pas une charge fixe
```

**Étape 3** : Vérifier régularité du jour
```python
# Jour moyen du mois
days_of_month = [tx.date.day for tx in sorted_txs]
avg_day = int(mean(days_of_month))
day_variance = stdev(days_of_month)

if day_variance > 5:
    return None  # Pas régulier
```

**Étape 4** : Vérifier intervalle mensuel
```python
# Calculer intervalles entre transactions
intervals = []
for i in range(1, len(sorted_txs)):
    delta_days = (sorted_txs[i].date - sorted_txs[i-1].date).days
    intervals.append(delta_days)

avg_interval = mean(intervals)

# Doit être proche de 30 jours (±10 jours)
if not (20 <= avg_interval <= 40):
    return None  # Pas mensuel
```

**Étape 5** : Calculer score de confiance
```python
# Facteur occurrences (max 0.4)
occurrence_score = min(occurrence_count / 6.0, 1.0) × 0.4

# Facteur variance montant (max 0.3)
amount_score = max(0, 1.0 - (amount_variance / 10.0)) × 0.3

# Facteur variance jour (max 0.2)
day_score = max(0, 1.0 - (day_variance / 5.0)) × 0.2

# Facteur régularité intervalle (max 0.1)
interval_score = max(0, 1.0 - abs(avg_interval - 30) / 10.0) × 0.1

confidence = occurrence_score + amount_score + day_score + interval_score
```

**Étape 6** : Accepter si confiance ≥ 70%
```python
if confidence >= 0.7:
    charge_fixe = {
        'merchant_name': merchant,
        'avg_amount': avg_amount,
        'recurrence_day': avg_day,
        'recurrence_confidence': confidence,
        'transaction_count': len(sorted_txs)
    }
    save_to_database(charge_fixe)
```

### Exemple de détection réussie

**Transactions Netflix** :
```
2024-01-01 : 21.60 EUR
2024-02-01 : 21.60 EUR
2024-03-01 : 21.60 EUR
2024-04-01 : 21.60 EUR
...
2024-12-01 : 21.60 EUR

→ Variance montant : 0%
→ Variance jour : 0
→ Intervalle moyen : 30.5 jours
→ Confiance : 96%
→ ✅ DÉTECTÉE comme charge fixe
```

---

## Classification des dépenses

### Vue d'ensemble

Toutes les dépenses sont classées en **3 catégories principales** :
1. **Charges fixes** : Obligatoires et récurrentes
2. **Charges semi-fixes** : Nécessaires mais ajustables
3. **Charges variables** : Discrétionnaires

### 1. Charges fixes (`fixed_charges_total`)

**Composition** :
```python
fixed_charges_total = charges_fixes_detectees + charges_fixes_structurelles
```

#### a) Charges fixes détectées automatiquement
Montant moyen des charges détectées par l'algorithme de récurrence.

**Exemple** :
```
Netflix : 21.60 EUR
Orange : 40.00 EUR
Assurance Utwin : 29.07 EUR
Allianz : 10.63 EUR
→ Total détecté : 101.30 EUR
```

#### b) Charges fixes structurelles
Montant moyen mensuel des catégories fixes identifiées dans les transactions.

**Catégories considérées comme fixes** :
```python
fixed_categories = [
    'prêt',
    'crédit',
    'assurance',
    'loyer',
    'bail',
    'pension',
    'garde',         # garde d'enfants
    'scolarité',
    'téléphone',
    'internet',
    'abonnement',
    'impôt',
    'taxe'
]
```

**Calcul** :
```python
# Pour chaque catégorie dans category_breakdown
for category, avg_amount in category_breakdown.items():
    category_lower = category.lower()

    if any(fixed in category_lower for fixed in fixed_categories):
        structural_fixed_total += avg_amount
```

**Exemple** :
```
Catégorie "prêt" : 1968.89 EUR/mois
Catégorie "Impôts" : 117.75 EUR/mois
Catégorie "Garde d'enfants" : 143.65 EUR/mois
Catégorie "Pension alimentaire" : 137.28 EUR/mois
...
→ Total structurel : 2657.38 EUR
```

**Total charges fixes** :
```
Détectées : 101.30 EUR
Structurelles : 2657.38 EUR
→ TOTAL : 2758.68 EUR
```

### 2. Charges semi-fixes (`semi_fixed_charges_total`)

**Définition** : Dépenses récurrentes et nécessaires, mais dont le montant peut varier et être ajusté.

**Catégories** :
```python
semi_fixed_categories = [
    'alimentation',
    'courses',
    'carburant',
    'transport',
    'santé',
    'pharmacie',
    'entretien',
    'électricité',
    'eau',
    'énergie',
    'essence',
    'garage'
]
```

**Calcul** :
```python
for category, avg_amount in category_breakdown.items():
    category_lower = category.lower()

    if any(semi in category_lower for semi in semi_fixed_categories):
        semi_fixed_total += avg_amount
```

**Exemple** :
```
Alimentation : 384.55 EUR/mois
Carburant : 56.96 EUR/mois
Électricité/eau : 170.68 EUR/mois
Entretien maison : 219.48 EUR/mois
Transport : 29.84 EUR/mois
Santé/pharmacie : 29.57 EUR/mois
Garage : 35.17 EUR/mois
→ TOTAL : 926.25 EUR
```

### 3. Charges variables (`variable_charges_total`)

**Définition** : Dépenses discrétionnaires, non obligatoires, facilement ajustables.

**Catégories** :
```python
variable_categories = [
    'loisirs',
    'restaurant',
    'shopping',
    'vêtement',
    'cadeau',
    'voyage',
    'divertissement',
    'streaming',
    'paris',
    'jeux',
    'loterie',
    'ligne'  # achats en ligne
]
```

**Calcul** :
```python
for category, avg_amount in category_breakdown.items():
    category_lower = category.lower()

    if any(var in category_lower for var in variable_categories):
        variable_total += avg_amount
    elif not (fixed or semi_fixed):
        # Toutes les autres catégories non classées
        other_total += avg_amount

# Ajouter "autres" aux variables
variable_total += other_total
```

**Exemple** :
```
Achats en ligne : 216.07 EUR/mois
Paris sportif : 76.74 EUR/mois
Vêtements : 48.71 EUR/mois
streaming : 25.20 EUR/mois
Loisirs : 22.78 EUR/mois
Restaurant : 17.45 EUR/mois
Jeux d'argent : 13.65 EUR/mois
Loterie : 3.73 EUR/mois
+ Autres non classées : 4153.19 EUR/mois
→ TOTAL : 4577.52 EUR
```

### Catégorie "Autres"

**Toutes les dépenses non classées** dans les catégories ci-dessus sont ajoutées aux charges variables :
- Virements sortants
- Chèques émis
- Retraits espèces
- Autres paiements
- Aide
- CAF
- Frais bancaires
- Services
- Amendes
- Remboursements
- etc.

**Justification** : Ces catégories sont considérées comme variables car :
1. Ce sont souvent des **moyens de paiement** (on ne connaît pas la destination réelle)
2. Par prudence, on les classe en **discrétionnaires** plutôt qu'obligatoires
3. L'utilisateur peut potentiellement les **réduire ou éviter**

---

## Métriques calculées

### 1. Reste à vivre (`remaining_to_live`)

**Définition** : Budget mensuel disponible après paiement des charges fixes.

**Calcul** :
```python
remaining_to_live = avg_monthly_income - fixed_charges_total
```

**Exemple** :
```
Revenus : 7113.63 EUR
Charges fixes : 756.98 EUR
→ Reste à vivre : 6356.65 EUR
```

**Interprétation** :
- **> 0** : L'utilisateur peut couvrir ses charges fixes
- **< 0** : Situation financière critique, revenus insuffisants pour les charges fixes

---

## Segmentation utilisateur

### Définition (`user_segment`)

Classification de la situation budgétaire de l'utilisateur.

### Critères

**Basé sur le ratio** :
```python
ratio = avg_monthly_expenses / avg_monthly_income
```

### Segments

| Segment | Ratio | Description |
|---------|-------|-------------|
| **budget_serré** | > 0.90 | Dépenses > 90% des revenus, peu de marge |
| **équilibré** | 0.70 - 0.90 | Situation saine, bon équilibre épargne/dépenses |
| **confortable** | < 0.70 | Revenus largement supérieurs aux dépenses |
| **indéterminé** | N/A | Revenus = 0 ou données insuffisantes |

### Calcul

```python
def determine_segment(avg_income, avg_expenses):
    if avg_income <= 0:
        return 'indéterminé'

    ratio = avg_expenses / avg_income

    if ratio > 0.90:
        return 'budget_serré'
    elif ratio >= 0.70:
        return 'équilibré'
    else:
        return 'confortable'
```

### Exemples

**Exemple 1** :
```
Revenus : 2500 EUR
Dépenses : 2400 EUR
Ratio : 2400 / 2500 = 0.96 (96%)
→ Segment : budget_serré
```

**Exemple 2** :
```
Revenus : 3000 EUR
Dépenses : 2400 EUR
Ratio : 2400 / 3000 = 0.80 (80%)
→ Segment : équilibré
```

**Exemple 3** :
```
Revenus : 4000 EUR
Dépenses : 2400 EUR
Ratio : 2400 / 4000 = 0.60 (60%)
→ Segment : confortable
```

---

## Pattern comportemental

### Définition (`behavioral_pattern`)

Identifie le comportement de dépense de l'utilisateur sur le dernier mois.

### Critères analysés

**Données** : Transactions du dernier mois uniquement

```python
# Filtrer débits du dernier mois
debits = [tx for tx in last_month_transactions if tx.amount < 0]

# Nombre de transactions par semaine
tx_per_week = len(debits) / 4.0

# Montant moyen par transaction
avg_tx_amount = sum(abs(tx.amount) for tx in debits) / len(debits)
```

### Patterns

| Pattern | Critères | Description |
|---------|----------|-------------|
| **acheteur_impulsif** | tx/semaine > 10 ET montant < 20 EUR | Nombreux petits achats fréquents |
| **planificateur** | tx/semaine < 5 ET montant > 50 EUR | Peu de transactions, montants élevés |
| **dépensier_hebdomadaire** | Autres cas | Comportement moyen, achats réguliers |
| **indéterminé** | Pas de données | Données insuffisantes |

### Calcul

```python
def determine_behavioral_pattern(user_id):
    # Récupérer transactions du dernier mois
    transactions = get_user_transactions(user_id, months_back=1)

    if not transactions:
        return 'indéterminé'

    debits = [tx for tx in transactions if tx.is_debit]

    if not debits:
        return 'indéterminé'

    # Métriques
    tx_per_week = len(debits) / 4.0
    avg_tx_amount = sum(abs(tx.amount) for tx in debits) / len(debits)

    # Classification
    if tx_per_week > 10 and avg_tx_amount < 20:
        return 'acheteur_impulsif'
    elif tx_per_week < 5 and avg_tx_amount > 50:
        return 'planificateur'
    else:
        return 'dépensier_hebdomadaire'
```

### Exemples

**Exemple 1 - Acheteur impulsif** :
```
Dernier mois : 48 débits
→ 12 transactions/semaine
Montant moyen : 15.50 EUR
→ Pattern : acheteur_impulsif
```

**Exemple 2 - Planificateur** :
```
Dernier mois : 15 débits
→ 3.75 transactions/semaine
Montant moyen : 120 EUR
→ Pattern : planificateur
```

**Exemple 3 - Dépensier hebdomadaire** :
```
Dernier mois : 28 débits
→ 7 transactions/semaine
Montant moyen : 35 EUR
→ Pattern : dépensier_hebdomadaire
```

---

## Complétude du profil

### Définition (`profile_completeness`)

Score entre **0.0 et 1.0** évaluant la qualité et la fiabilité du profil calculé.

### Facteurs de calcul

Le score est basé sur **3 facteurs** :

#### 1. Facteur mois de données (max 0.4)

**Calcul** :
```python
if months_required is None:
    # Analyse complète : score basé sur nombre absolu
    # 12+ mois = score complet (0.4)
    months_score = min(nb_months_reels / 12.0, 1.0) × 0.4
else:
    # Analyse paramétrée : score basé sur ratio
    months_score = min(nb_months_reels / months_required, 1.0) × 0.4
```

**Exemples** :
```
60 mois disponibles, analyse complète (None)
→ min(60 / 12, 1.0) × 0.4 = 1.0 × 0.4 = 0.40

3 mois disponibles, analyse 12 mois demandée
→ min(3 / 12, 1.0) × 0.4 = 0.25 × 0.4 = 0.10

12 mois disponibles, analyse 12 mois demandée
→ min(12 / 12, 1.0) × 0.4 = 1.0 × 0.4 = 0.40
```

#### 2. Facteur charges fixes détectées (max 0.3)

**Calcul** :
```python
# 5+ charges fixes = score complet (0.3)
fixed_charges_score = min(nb_charges_fixes / 5.0, 1.0) × 0.3
```

**Exemples** :
```
4 charges fixes détectées
→ min(4 / 5, 1.0) × 0.3 = 0.8 × 0.3 = 0.24

7 charges fixes détectées
→ min(7 / 5, 1.0) × 0.3 = 1.0 × 0.3 = 0.30
```

#### 3. Facteur présence de revenus (max 0.3)

**Calcul** :
```python
has_income = any(month.total_income > 0 for month in monthly_aggregates)
income_score = 0.3 if has_income else 0.0
```

**Exemples** :
```
Au moins un mois avec revenus > 0
→ 0.30

Aucun revenu détecté
→ 0.00
```

### Score final

```python
profile_completeness = months_score + fixed_charges_score + income_score
profile_completeness = min(max(profile_completeness, 0.0), 1.0)  # Limiter [0.0, 1.0]
```

### Exemples complets

**Exemple 1 - Profil excellent** :
```
60 mois de données (analyse complète)
→ months_score = 0.40

7 charges fixes détectées
→ fixed_charges_score = 0.30

Revenus présents
→ income_score = 0.30

TOTAL : 0.40 + 0.30 + 0.30 = 1.00 (100%)
```

**Exemple 2 - Profil moyen** :
```
6 mois de données (analyse 12 mois)
→ months_score = 0.20

2 charges fixes détectées
→ fixed_charges_score = 0.12

Revenus présents
→ income_score = 0.30

TOTAL : 0.20 + 0.12 + 0.30 = 0.62 (62%)
```

**Exemple 3 - Profil faible** :
```
2 mois de données (analyse complète)
→ months_score = 0.07

0 charges fixes détectées
→ fixed_charges_score = 0.00

Pas de revenus
→ income_score = 0.00

TOTAL : 0.07 + 0.00 + 0.00 = 0.07 (7%)
```

### Interprétation

| Score | Qualité | Recommandation |
|-------|---------|----------------|
| 0.90 - 1.00 | Excellente | Profil très fiable, utilisable pour toutes analyses |
| 0.70 - 0.89 | Bonne | Profil fiable, quelques données supplémentaires seraient utiles |
| 0.50 - 0.69 | Moyenne | Profil utilisable mais limité, encourager plus d'historique |
| 0.30 - 0.49 | Faible | Profil peu fiable, données insuffisantes |
| 0.00 - 0.29 | Très faible | Profil non fiable, ne pas utiliser pour prédictions |

---

## Résumé du flux de calcul

```
1. Récupération transactions
   ├─ Filtrer par user_id, période, deleted=false
   └─ Trier par date

2. Calcul des agrégats mensuels
   ├─ Grouper par mois (YYYY-MM)
   ├─ Pour chaque mois :
   │  ├─ total_income = sum(crédits)
   │  ├─ total_expenses = sum(|débits|)
   │  └─ net_cashflow = income - expenses
   └─ Calculer moyennes sur tous les mois

3. Détection charges fixes
   ├─ Grouper débits par marchand
   ├─ Pour chaque marchand :
   │  ├─ Analyser récurrence temporelle
   │  ├─ Analyser variance montant
   │  └─ Calculer score confiance
   └─ Sauvegarder si confiance ≥ 70%

4. Répartition catégories
   ├─ Calculer breakdown par catégorie (moyennes mensuelles)
   ├─ Classifier en :
   │  ├─ Charges fixes structurelles
   │  ├─ Charges semi-fixes
   │  └─ Charges variables (incluant "autres")
   └─ Total charges fixes = détectées + structurelles

5. Métriques finales
   ├─ Reste à vivre = revenus - charges fixes
   ├─ Taux épargne = (revenus - dépenses) / revenus × 100
   ├─ Segment = f(ratio dépenses/revenus)
   ├─ Pattern = f(fréquence, montant moyen dernier mois)
   └─ Complétude = f(nb_mois, nb_charges_fixes, has_revenus)

6. Sauvegarde profil
   └─ Table : user_budget_profiles
```

---

## Endpoints API

### GET `/api/v1/budget/profile`
Récupère le profil budgétaire sauvegardé (dernière analyse).

### POST `/api/v1/budget/profile/analyze`
Calcule et sauvegarde un nouveau profil budgétaire.

**Body** :
```json
{
  "months_analysis": 12  // optionnel, null = toutes transactions
}
```

**Réponse** :
```json
{
  "user_segment": "budget_serré",
  "behavioral_pattern": "dépensier_hebdomadaire",
  "avg_monthly_income": 7113.63,
  "avg_monthly_expenses": 7336.10,
  "avg_monthly_savings": -222.47,
  "savings_rate": -3.13,
  "fixed_charges_total": 756.98,
  "semi_fixed_charges_total": 1096.06,
  "variable_charges_total": 5584.36,
  "remaining_to_live": 6356.64,
  "profile_completeness": 0.94,
  "last_analyzed_at": "2025-10-19T06:38:53+00:00"
}
```

### GET `/api/v1/budget/monthly-aggregates?months=12`
Récupère les agrégats mensuels (revenus, dépenses par mois).

### GET `/api/v1/budget/category-breakdown?months=12`
Récupère la répartition moyenne mensuelle par catégorie.

### GET `/api/v1/budget/fixed-charges`
Récupère la liste des charges fixes détectées automatiquement.

---

## Fichiers sources

| Fichier | Rôle |
|---------|------|
| `services/budget_profiler.py` | Service principal de calcul du profil |
| `services/transaction_service.py` | Récupération et agrégation des transactions |
| `services/fixed_charge_detector.py` | Détection automatique des charges fixes |
| `api/routes/budget_profile.py` | Endpoints API |
| `models/budget_profiling.py` | Modèles SQLAlchemy |

---

## Changelog

### Version 1.1.0 (2025-10-19)
- ✅ Correction calcul moyennes mensuelles (était en totaux cumulés)
- ✅ Correction calcul nombre de mois (cohérence aggregates/breakdown)
- ✅ Ajout charges fixes structurelles (prêt, impôts, etc.)
- ✅ Enrichissement classification catégories
- ✅ Toutes dépenses non classées comptées dans variables

### Version 1.0.0
- Version initiale

---

**Document généré le** : 2025-10-19
**Service** : Budget Profiling Service v1.1.0
**Auteur** : Claude Code
