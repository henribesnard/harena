# Analytics Agent Documentation

## Vue d'ensemble

L'Analytics Agent est un agent logique (sans appels LLM) qui effectue des analyses statistiques avancées sur les transactions financières.

## Fonctionnalités

### 1. Comparaisons Temporelles (`compare_periods`)

Compare deux périodes de transactions (MoM, YoY, QoQ, WoW).

**Utilisation :**

```python
from conversation_service.agents.logic.analytics_agent import AnalyticsAgent, PeriodType

agent = AnalyticsAgent()

# Comparer janvier 2025 vs décembre 2024
result = await agent.compare_periods(
    transactions_period_1=transactions_january,
    transactions_period_2=transactions_december,
    period_type=PeriodType.MOM,
    top_contributors_limit=5
)

print(f"Changement: {result.percentage_change}%")
print(f"Top contributeurs: {result.top_contributors}")
```

**Résultat :**

```python
ComparisonResult(
    period_1_total=11523.45,
    period_2_total=9219.34,
    absolute_change=2304.11,
    percentage_change=25.0,
    period_1_count=93,
    period_2_count=93,
    period_1_average=123.91,
    period_2_average=99.13,
    top_contributors=[
        {
            'name': 'Alimentation',
            'period_1_amount': 5000.0,
            'period_2_amount': 4000.0,
            'absolute_change': 1000.0,
            'percentage_change': 25.0
        }
    ],
    period_type='month_over_month'
)
```

### 2. Analyse de Tendances (`calculate_trend`)

Calcule la tendance avec régression linéaire et prévisions.

**Utilisation :**

```python
# Analyser tendance des 3 derniers mois
result = await agent.calculate_trend(
    transactions=transactions_last_3_months,
    aggregation="monthly",  # "daily", "weekly", "monthly"
    forecast_periods=2
)

print(f"Direction: {result.direction}")
print(f"Pente: {result.slope}")
print(f"R²: {result.r_squared}")
print(f"Prévisions: {result.forecast_values}")
```

**Résultat :**

```python
TrendAnalysis(
    direction=TrendDirection.INCREASING,
    slope=154.74,
    intercept=9000.0,
    r_squared=0.381,
    forecast_values=[9557.01, 9711.76],
    forecast_periods=['2025-02-28', '2025-03-31'],
    confidence_interval_95=([9400.0, 9550.0], [9700.0, 9850.0])
)
```

### 3. Détection d'Anomalies (`detect_anomalies`)

Détecte les transactions anormales avec Z-score ou IQR.

**Utilisation :**

```python
from conversation_service.agents.logic.analytics_agent import AnomalyMethod

# Détecter anomalies avec Z-score
anomalies = await agent.detect_anomalies(
    transactions=transactions,
    method=AnomalyMethod.ZSCORE,
    threshold=2.0  # 2 écarts-types
)

for anomaly in anomalies:
    print(f"{anomaly.merchant_name}: {anomaly.amount}€ - {anomaly.reason}")
```

**Résultat :**

```python
[
    Anomaly(
        transaction_id='12345',
        amount=1500.0,
        date='2025-01-15',
        merchant_name='Tesla',
        score=3.2,
        reason='Amount is 3.2σ from mean (mean=150.00, std=100.00)',
        severity='high'
    )
]
```

### 4. Analyse Pivot (`pivot_analysis`)

Génère des tables pivots multi-dimensionnelles.

**Utilisation :**

```python
# Analyser dépenses par catégorie × mois
result = await agent.pivot_analysis(
    transactions=transactions,
    rows="category",
    columns="month",
    values="amount",
    aggfunc="sum"
)

print(result.data)  # DataFrame pandas
print(f"Total: {result.grand_total}")
```

**Résultat :**

```python
PivotTable(
    data=DataFrame(
        # category    2024-11    2024-12    2025-01
        # Alimentation  3000.0    3500.0    4000.0
        # Transport     1000.0    1200.0    1300.0
    ),
    row_totals=Series(...),
    column_totals=Series(...),
    grand_total=13000.0
)
```

### 5. Résumé Métriques (`calculate_metrics_summary`)

Calcule un résumé statistique complet.

**Utilisation :**

```python
summary = await agent.calculate_metrics_summary(transactions)

print(f"Total: {summary['total']}")
print(f"Moyenne: {summary['average']}")
print(f"Médiane: {summary['median']}")
print(f"Écart-type: {summary['std_dev']}")
```

## Cas d'Usage par Question

### Q1: "Compare mes dépenses de ce mois avec le mois dernier"

```python
result = await agent.compare_periods(
    transactions_current_month,
    transactions_previous_month,
    PeriodType.MOM
)
# → result.percentage_change, result.top_contributors
```

### Q2: "Différence revenus année vs année dernière"

```python
result = await agent.compare_periods(
    transactions_2025,
    transactions_2024,
    PeriodType.YOY
)
# → result.absolute_change, result.percentage_change
```

### Q3: "Evolution dépenses sur 3 derniers mois"

```python
result = await agent.calculate_trend(
    transactions_3_months,
    aggregation="monthly",
    forecast_periods=1
)
# → result.direction, result.slope, result.forecast_values
```

### Q4: "Dépenses alimentaires YoY"

```python
# Filtrer par catégorie avant
transactions_food_2025 = filter_by_category(transactions_2025, "Alimentation")
transactions_food_2024 = filter_by_category(transactions_2024, "Alimentation")

result = await agent.compare_periods(
    transactions_food_2025,
    transactions_food_2024,
    PeriodType.YOY
)
```

### Q5: "Mois de plus forte dépense sur 12 mois"

```python
result = await agent.pivot_analysis(
    transactions_12_months,
    rows="category",
    columns="month",
    values="amount",
    aggfunc="sum"
)

peak_month = result.column_totals.idxmax()
peak_amount = result.column_totals.max()
# → peak_month, peak_amount
```

## Types de Périodes Supportés

```python
class PeriodType(str, Enum):
    MOM = "month_over_month"    # Mois sur mois
    YOY = "year_over_year"      # Année sur année
    QOQ = "quarter_over_quarter" # Trimestre sur trimestre
    WOW = "week_over_week"      # Semaine sur semaine
```

## Méthodes de Détection d'Anomalies

```python
class AnomalyMethod(str, Enum):
    ZSCORE = "zscore"  # Basé sur l'écart-type (distribution normale)
    IQR = "iqr"        # Basé sur l'écart interquartile (robuste aux outliers)
```

## Performance

| Fonction | Temps Moyen | Complexité |
|----------|-------------|------------|
| `compare_periods` | ~10-50ms | O(n) |
| `calculate_trend` | ~20-100ms | O(n log n) |
| `detect_anomalies` | ~10-50ms | O(n) |
| `pivot_analysis` | ~50-200ms | O(n log n) |

où n = nombre de transactions

## Tests

Exécuter la suite de tests :

```bash
python scripts/test_phase1_analytics.py
```

Résultats attendus : 5/5 tests passés (Q1-Q5)

## Intégration avec Response Generator

L'Analytics Agent sera appelé automatiquement par le Response Generator lorsque des métriques avancées sont requises :

```python
# Dans response_generator.py
from conversation_service.agents.logic.analytics_agent import AnalyticsAgent

agent = AnalyticsAgent()

# Détecter si comparaison temporelle nécessaire
if intent requires comparison:
    comparison = await agent.compare_periods(...)
    # Enrichir contexte LLM avec métriques
```

## Roadmap

### Phase 1 ✅ (Complété)
- Comparaisons MoM/YoY/QoQ/WoW
- Analyse tendances avec régression linéaire
- Détection anomalies Z-score/IQR
- Tables pivots multi-dimensionnelles

### Phase 2 (À venir)
- Forecasting ML (ARIMA, Prophet)
- Clustering catégories
- Détection patterns récurrents
- Analyse saisonnalité

## Dépendances

```txt
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0
```

Installer avec :

```bash
pip install pandas numpy scipy
```

## Support

Pour questions ou bugs, créer une issue dans le repo ou contacter l'équipe data.

## Changelog

### v1.0.0 (2025-01-12)
- ✅ Implémentation initiale Analytics Agent
- ✅ 5 fonctions principales
- ✅ Tests Q1-Q5 passent (5/5)
- ✅ Documentation complète
