# Sprint 1.3 - Visualisations de Base - Plan Détaillé

**Date début**: 2025-10-12
**Durée estimée**: 1 semaine (5 jours)
**Branche**: `feature/phase1-visualizations`
**Tag cible**: `v3.2.6.3`
**Baseline stable**: v3.2.6.2 (Sprint 1.2 validé)

---

## 🎯 Objectifs du Sprint 1.3

Implémenter un système de génération de spécifications de visualisations (graphiques, courbes, KPI cards) pour enrichir les réponses avec des données visuelles.

### Fonctionnalités Principales

1. **Système de Génération de Specs Visualisations**
   - Architecture modulaire pour différents types de visualisations
   - Mapping automatique intent → type de visualisation
   - Génération specs Chart.js compatibles frontend

2. **Types de Visualisations Supportés**
   - KPI Cards (totaux, moyennes, comparaisons)
   - Line Charts (évolutions temporelles, tendances)
   - Bar Charts (catégories, comparaisons)
   - Pie Charts (répartitions, pourcentages)

3. **Intégration Response Generator**
   - Détection automatique du besoin de visualisation
   - Génération specs basée sur les données
   - Format JSON prêt pour le frontend

---

## 📋 Tâches Détaillées

### T3.1 - Système de Génération Specs Visualisations (2 jours)

**Objectif**: Architecture de base pour la génération de specs de visualisations

#### Sous-tâches

1. **Créer modèles de données** `conversation_service/models/visualization/schemas.py`

```python
"""
Visualization Schemas - Data Models for Chart Specifications

Supports Chart.js format for frontend rendering
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class VisualizationType(str, Enum):
    """Types de visualisations supportés"""
    KPI_CARD = "kpi_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    DOUGHNUT_CHART = "doughnut_chart"
    AREA_CHART = "area_chart"


class KPICard(BaseModel):
    """KPI Card specification"""
    type: Literal["kpi_card"] = "kpi_card"
    title: str
    value: float
    unit: str = ""
    change_percent: Optional[float] = None
    change_direction: Optional[Literal["up", "down", "stable"]] = None
    comparison_text: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


class ChartDataset(BaseModel):
    """Dataset pour un graphique Chart.js"""
    label: str
    data: List[float]
    backgroundColor: Optional[List[str] | str] = None
    borderColor: Optional[str] = None
    borderWidth: Optional[int] = 2
    fill: Optional[bool] = False


class ChartData(BaseModel):
    """Données pour Chart.js"""
    labels: List[str]
    datasets: List[ChartDataset]


class ChartOptions(BaseModel):
    """Options pour Chart.js"""
    responsive: bool = True
    maintainAspectRatio: bool = False
    plugins: Dict[str, Any] = Field(default_factory=dict)
    scales: Optional[Dict[str, Any]] = None


class ChartVisualization(BaseModel):
    """Spécification complète d'un graphique Chart.js"""
    type: VisualizationType
    title: str
    description: Optional[str] = None
    data: ChartData
    options: ChartOptions = Field(default_factory=ChartOptions)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VisualizationResponse(BaseModel):
    """Réponse complète avec visualisations"""
    visualizations: List[KPICard | ChartVisualization]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    intent_group: str
    intent_subtype: str


__all__ = [
    "VisualizationType",
    "KPICard",
    "ChartDataset",
    "ChartData",
    "ChartOptions",
    "ChartVisualization",
    "VisualizationResponse"
]
```

2. **Créer service de base** `conversation_service/services/visualization/visualization_service.py`

```python
"""
Visualization Service - Generate Chart.js Specifications

Automatically generates visualizations based on:
- Intent type
- Search results
- Aggregations
- User preferences
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from conversation_service.models.visualization.schemas import (
    VisualizationType,
    KPICard,
    ChartVisualization,
    ChartData,
    ChartDataset,
    ChartOptions,
    VisualizationResponse
)

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for generating visualization specifications"""

    def __init__(self):
        # Mapping intent → types de visualisations
        self.intent_visualization_map = {
            "transaction_search.simple": [VisualizationType.KPI_CARD],
            "transaction_search.by_category": [
                VisualizationType.KPI_CARD,
                VisualizationType.PIE_CHART
            ],
            "analytics.comparison": [
                VisualizationType.KPI_CARD,
                VisualizationType.BAR_CHART
            ],
            "analytics.trend": [
                VisualizationType.KPI_CARD,
                VisualizationType.LINE_CHART
            ],
            "analytics.mom": [
                VisualizationType.KPI_CARD,
                VisualizationType.BAR_CHART
            ],
        }

    def generate_visualizations(
        self,
        intent_group: str,
        intent_subtype: str,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> VisualizationResponse:
        """
        Generate visualizations based on intent and data

        Args:
            intent_group: Intent group (e.g., "transaction_search")
            intent_subtype: Intent subtype (e.g., "by_category")
            search_results: Transaction data
            aggregations: Aggregated data from search
            user_preferences: User visualization preferences

        Returns:
            VisualizationResponse with specs for frontend
        """
        intent_key = f"{intent_group}.{intent_subtype}"
        viz_types = self.intent_visualization_map.get(
            intent_key,
            [VisualizationType.KPI_CARD]
        )

        visualizations = []

        for viz_type in viz_types:
            if viz_type == VisualizationType.KPI_CARD:
                kpis = self._generate_kpi_cards(
                    search_results,
                    aggregations,
                    intent_group,
                    intent_subtype
                )
                visualizations.extend(kpis)

            elif viz_type == VisualizationType.PIE_CHART:
                pie = self._generate_pie_chart(
                    search_results,
                    aggregations,
                    intent_group,
                    intent_subtype
                )
                if pie:
                    visualizations.append(pie)

            elif viz_type == VisualizationType.BAR_CHART:
                bar = self._generate_bar_chart(
                    search_results,
                    aggregations,
                    intent_group,
                    intent_subtype
                )
                if bar:
                    visualizations.append(bar)

            elif viz_type == VisualizationType.LINE_CHART:
                line = self._generate_line_chart(
                    search_results,
                    aggregations,
                    intent_group,
                    intent_subtype
                )
                if line:
                    visualizations.append(line)

        return VisualizationResponse(
            visualizations=visualizations,
            intent_group=intent_group,
            intent_subtype=intent_subtype
        )

    def _generate_kpi_cards(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> List[KPICard]:
        """Generate KPI cards from data"""
        # Implementation in T3.2
        pass

    def _generate_pie_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """Generate pie chart spec"""
        # Implementation in T3.2
        pass

    def _generate_bar_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """Generate bar chart spec"""
        # Implementation in T3.2
        pass

    def _generate_line_chart(
        self,
        search_results: List[Dict[str, Any]],
        aggregations: Optional[Dict[str, Any]],
        intent_group: str,
        intent_subtype: str
    ) -> Optional[ChartVisualization]:
        """Generate line chart spec"""
        # Implementation in T3.2
        pass


__all__ = ['VisualizationService']
```

**Critères d'acceptation**:
- ✅ Modèles Pydantic définis
- ✅ Architecture service créée
- ✅ Mapping intent → visualisations
- ✅ Format Chart.js compatible

---

### T3.2 - Générateur Chart.js Specs (2 jours)

**Objectif**: Implémenter la génération concrète des specs Chart.js

#### Implémentations des générateurs

1. **KPI Cards Generator**

```python
def _generate_kpi_cards(
    self,
    search_results: List[Dict[str, Any]],
    aggregations: Optional[Dict[str, Any]],
    intent_group: str,
    intent_subtype: str
) -> List[KPICard]:
    """
    Generate KPI cards from transaction data

    Examples:
    - Total spending
    - Transaction count
    - Average transaction amount
    - Comparison vs previous period
    """
    kpis = []

    if not search_results:
        return kpis

    # Total spending
    total_spending = sum(
        abs(tx.get('amount', 0))
        for tx in search_results
        if tx.get('transaction_type') == 'debit'
    )

    kpis.append(KPICard(
        title="Total Dépenses",
        value=total_spending,
        unit="€",
        icon="credit-card",
        color="red"
    ))

    # Transaction count
    kpis.append(KPICard(
        title="Nombre de Transactions",
        value=len(search_results),
        unit="transactions",
        icon="list",
        color="blue"
    ))

    # Average transaction
    if search_results:
        avg_amount = total_spending / len(search_results)
        kpis.append(KPICard(
            title="Montant Moyen",
            value=avg_amount,
            unit="€",
            icon="trending-up",
            color="green"
        ))

    # Comparison if available in aggregations
    if aggregations and 'comparison' in aggregations:
        comparison = aggregations['comparison']
        change_percent = comparison.get('change_percent', 0)
        direction = "up" if change_percent > 0 else "down" if change_percent < 0 else "stable"

        kpis[0].change_percent = abs(change_percent)
        kpis[0].change_direction = direction
        kpis[0].comparison_text = comparison.get('period_text', 'vs mois précédent')

    return kpis
```

2. **Pie Chart Generator**

```python
def _generate_pie_chart(
    self,
    search_results: List[Dict[str, Any]],
    aggregations: Optional[Dict[str, Any]],
    intent_group: str,
    intent_subtype: str
) -> Optional[ChartVisualization]:
    """
    Generate pie chart for category breakdown

    Shows spending distribution by category
    """
    # Group by category
    category_totals = {}
    for tx in search_results:
        category = tx.get('category', 'Autre')
        amount = abs(tx.get('amount', 0))
        category_totals[category] = category_totals.get(category, 0) + amount

    if not category_totals:
        return None

    # Sort by amount descending
    sorted_categories = sorted(
        category_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 5 categories + "Autres"
    top_categories = sorted_categories[:5]
    other_amount = sum(amount for cat, amount in sorted_categories[5:])

    labels = [cat for cat, _ in top_categories]
    values = [amount for _, amount in top_categories]

    if other_amount > 0:
        labels.append("Autres")
        values.append(other_amount)

    # Colors
    colors = [
        "#FF6384", "#36A2EB", "#FFCE56",
        "#4BC0C0", "#9966FF", "#FF9F40"
    ]

    return ChartVisualization(
        type=VisualizationType.PIE_CHART,
        title="Répartition par Catégorie",
        description=f"Top {len(top_categories)} catégories de dépenses",
        data=ChartData(
            labels=labels,
            datasets=[ChartDataset(
                label="Montant",
                data=values,
                backgroundColor=colors[:len(values)]
            )]
        ),
        options=ChartOptions(
            plugins={
                "legend": {"position": "right"},
                "tooltip": {
                    "callbacks": {
                        "label": "function(context) { return context.label + ': ' + context.parsed + '€'; }"
                    }
                }
            }
        )
    )
```

3. **Bar Chart Generator**

```python
def _generate_bar_chart(
    self,
    search_results: List[Dict[str, Any]],
    aggregations: Optional[Dict[str, Any]],
    intent_group: str,
    intent_subtype: str
) -> Optional[ChartVisualization]:
    """
    Generate bar chart for comparisons

    Examples:
    - Month-over-month comparison
    - Category comparison
    - Year-over-year comparison
    """
    if not aggregations or 'comparison_data' not in aggregations:
        return None

    comparison_data = aggregations['comparison_data']

    labels = comparison_data.get('labels', [])
    current_values = comparison_data.get('current', [])
    previous_values = comparison_data.get('previous', [])

    datasets = [
        ChartDataset(
            label="Période Actuelle",
            data=current_values,
            backgroundColor="#36A2EB",
            borderColor="#36A2EB"
        )
    ]

    if previous_values:
        datasets.append(ChartDataset(
            label="Période Précédente",
            data=previous_values,
            backgroundColor="#FF6384",
            borderColor="#FF6384"
        ))

    return ChartVisualization(
        type=VisualizationType.BAR_CHART,
        title="Comparaison des Dépenses",
        description="Évolution par rapport à la période précédente",
        data=ChartData(
            labels=labels,
            datasets=datasets
        ),
        options=ChartOptions(
            scales={
                "y": {
                    "beginAtZero": True,
                    "ticks": {
                        "callback": "function(value) { return value + '€'; }"
                    }
                }
            },
            plugins={
                "legend": {"position": "top"}
            }
        )
    )
```

4. **Line Chart Generator**

```python
def _generate_line_chart(
    self,
    search_results: List[Dict[str, Any]],
    aggregations: Optional[Dict[str, Any]],
    intent_group: str,
    intent_subtype: str
) -> Optional[ChartVisualization]:
    """
    Generate line chart for trends

    Shows spending evolution over time
    """
    if not aggregations or 'trend_data' not in aggregations:
        # Generate from search_results if available
        return self._generate_line_from_transactions(search_results)

    trend_data = aggregations['trend_data']

    labels = trend_data.get('periods', [])
    values = trend_data.get('values', [])

    return ChartVisualization(
        type=VisualizationType.LINE_CHART,
        title="Évolution des Dépenses",
        description="Tendance sur les derniers mois",
        data=ChartData(
            labels=labels,
            datasets=[ChartDataset(
                label="Dépenses",
                data=values,
                borderColor="#36A2EB",
                backgroundColor="rgba(54, 162, 235, 0.1)",
                fill=True,
                tension=0.4
            )]
        ),
        options=ChartOptions(
            scales={
                "y": {
                    "beginAtZero": True,
                    "ticks": {
                        "callback": "function(value) { return value + '€'; }"
                    }
                }
            },
            plugins={
                "legend": {"display": False}
            }
        )
    )
```

**Critères d'acceptation**:
- ✅ 4 générateurs implémentés (KPI, Pie, Bar, Line)
- ✅ Specs Chart.js valides
- ✅ Gestion données manquantes
- ✅ Colors et styling cohérents

---

### T3.3 - Intégration Response Generator (1 jour)

**Objectif**: Intégrer VisualizationService dans le Response Generator

#### Modifications Response Generator

```python
class ResponseGenerator:
    def __init__(self, ...):
        # ... existing code ...

        # Sprint 1.3 - Visualization Service
        self.visualization_service = VisualizationService()
        self.enable_visualizations = enable_visualizations

    async def generate_response(
        self,
        request: ResponseGenerationRequest
    ) -> GeneratedResponse:
        """Generate response with visualizations"""

        # ... existing response generation ...

        # Generate visualizations if enabled
        visualizations = []
        if self.enable_visualizations and request.search_results:
            try:
                viz_response = self.visualization_service.generate_visualizations(
                    intent_group=request.intent_group,
                    intent_subtype=request.intent_subtype,
                    search_results=request.search_results,
                    aggregations=request.search_aggregations,
                    user_preferences=request.user_profile
                )
                visualizations = viz_response.visualizations

                logger.info(f"Generated {len(visualizations)} visualizations")

            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}")
                # Graceful degradation - continue without viz

        return GeneratedResponse(
            response_text=response_text,
            insights=insights,
            visualizations=visualizations,  # NEW
            # ... rest of response ...
        )
```

**Critères d'acceptation**:
- ✅ VisualizationService intégré
- ✅ Visualizations dans GeneratedResponse
- ✅ Graceful degradation si erreur
- ✅ Logging approprié

---

### T3.4 - Tests E2E Visualisations (1 jour)

**Objectif**: Valider bout-en-bout la génération de visualisations

#### Tests E2E

```python
"""
E2E Tests - Sprint 1.3: Visualizations

Tests complets pour la génération de visualisations
"""

import pytest
from conversation_service.services.visualization.visualization_service import VisualizationService
from conversation_service.models.visualization.schemas import VisualizationType


@pytest.mark.e2e
def test_e2e_kpi_cards_generation():
    """Test génération KPI cards depuis transactions"""

    service = VisualizationService()

    transactions = [
        {'amount': -100, 'transaction_type': 'debit', 'category': 'Restaurant'},
        {'amount': -150, 'transaction_type': 'debit', 'category': 'Transport'},
        {'amount': -80, 'transaction_type': 'debit', 'category': 'Restaurant'},
    ]

    response = service.generate_visualizations(
        intent_group="transaction_search",
        intent_subtype="simple",
        search_results=transactions
    )

    # Validations
    assert len(response.visualizations) >= 2

    kpi_cards = [v for v in response.visualizations if v.type == "kpi_card"]
    assert len(kpi_cards) >= 2

    # Total spending KPI
    total_kpi = kpi_cards[0]
    assert total_kpi.title == "Total Dépenses"
    assert total_kpi.value == 330.0
    assert total_kpi.unit == "€"


@pytest.mark.e2e
def test_e2e_pie_chart_category_breakdown():
    """Test génération pie chart pour catégories"""

    service = VisualizationService()

    transactions = [
        {'amount': -100, 'category': 'Restaurant'},
        {'amount': -150, 'category': 'Transport'},
        {'amount': -80, 'category': 'Restaurant'},
        {'amount': -50, 'category': 'Shopping'},
    ]

    response = service.generate_visualizations(
        intent_group="transaction_search",
        intent_subtype="by_category",
        search_results=transactions
    )

    # Trouver le pie chart
    pie_charts = [v for v in response.visualizations if v.type == VisualizationType.PIE_CHART]
    assert len(pie_charts) == 1

    pie = pie_charts[0]
    assert pie.title == "Répartition par Catégorie"
    assert len(pie.data.labels) == 3
    assert "Restaurant" in pie.data.labels
    assert "Transport" in pie.data.labels


@pytest.mark.e2e
def test_e2e_bar_chart_comparison():
    """Test génération bar chart pour comparaison"""

    service = VisualizationService()

    aggregations = {
        'comparison_data': {
            'labels': ['Jan', 'Fev', 'Mar'],
            'current': [1000, 1200, 1100],
            'previous': [950, 1150, 1050]
        }
    }

    response = service.generate_visualizations(
        intent_group="analytics",
        intent_subtype="comparison",
        search_results=[],
        aggregations=aggregations
    )

    bar_charts = [v for v in response.visualizations if v.type == VisualizationType.BAR_CHART]
    assert len(bar_charts) == 1

    bar = bar_charts[0]
    assert len(bar.data.datasets) == 2  # Current + Previous
    assert bar.data.datasets[0].label == "Période Actuelle"
    assert bar.data.datasets[1].label == "Période Précédente"


@pytest.mark.e2e
def test_e2e_line_chart_trend():
    """Test génération line chart pour tendance"""

    service = VisualizationService()

    aggregations = {
        'trend_data': {
            'periods': ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun'],
            'values': [1000, 1100, 1050, 1200, 1150, 1300]
        }
    }

    response = service.generate_visualizations(
        intent_group="analytics",
        intent_subtype="trend",
        search_results=[],
        aggregations=aggregations
    )

    line_charts = [v for v in response.visualizations if v.type == VisualizationType.LINE_CHART]
    assert len(line_charts) == 1

    line = line_charts[0]
    assert len(line.data.labels) == 6
    assert line.data.datasets[0].fill is True  # Area chart


@pytest.mark.e2e
def test_e2e_visualization_graceful_degradation():
    """Test graceful degradation si données insuffisantes"""

    service = VisualizationService()

    # Pas de transactions
    response = service.generate_visualizations(
        intent_group="transaction_search",
        intent_subtype="simple",
        search_results=[]
    )

    # Devrait retourner réponse vide, pas d'erreur
    assert response is not None
    assert len(response.visualizations) == 0


@pytest.mark.e2e
def test_e2e_response_generator_with_visualizations():
    """Test integration complète Response Generator + Visualizations"""

    # This test requires full Response Generator setup
    # To be implemented with real integration
    pass
```

**Critères d'acceptation**:
- ✅ Tests pour chaque type de visualisation
- ✅ Tests graceful degradation
- ✅ Tests intégration Response Generator
- ✅ >85% coverage

---

## 🛡️ Plan de Rollback Sprint 1.3

### Rollback Code

```bash
# Retour vers v3.2.6.2 (Sprint 1.2)
git checkout v3.2.6.2

# Redéploiement
./scripts/deploy_production.sh --tag v3.2.6.2 --force

# Vérifier health
curl https://api.harena.com/health
```

**Note**: Pas de migration database pour Sprint 1.3, donc rollback code uniquement.

---

## 📊 Critères d'Acceptation Globaux

| Critère | Objectif | Mesure |
|---------|---------|--------|
| **Tests** | >85% coverage | pytest --cov |
| **Visualisations générées** | 100% des intents supportés | Tests E2E |
| **Format Chart.js** | Valide | Validation schemas |
| **Graceful degradation** | Aucun crash si erreur | Tests erreurs |
| **Régression** | Aucune vs v3.2.6.2 | Tests régression |
| **Performance** | <50ms génération viz | Benchmarks |

---

## 🚀 Déploiement Sprint 1.3

### Séquence Déploiement

1. **Validation locale** ✅
   ```bash
   pytest tests/ -v
   ```

2. **Staging deployment** ✅
   ```bash
   ./scripts/deploy_staging.sh --tag v3.2.6.3-rc1
   # Monitoring 24h
   ```

3. **Production deployment** ✅
   ```bash
   # Canary 10% → 50% → 100%
   ./scripts/deploy_canary.sh --from v3.2.6.2 --to v3.2.6.3 --percentage 10
   ```

4. **Tag final** ✅
   ```bash
   git tag -a v3.2.6.3 -m "Sprint 1.3 Complete: Visualizations"
   git push origin v3.2.6.3
   ```

---

## 🎯 Timeline Estimée

| Tâche | Durée | Début | Fin |
|-------|-------|-------|-----|
| T3.1: Système génération specs | 2 jours | J1 | J2 |
| T3.2: Générateur Chart.js | 2 jours | J2 | J3 |
| T3.3: Intégration Response Generator | 1 jour | J4 | J4 |
| T3.4: Tests E2E | 1 jour | J5 | J5 |
| **TOTAL** | **5 jours** | | **1 semaine** |

---

## 🎉 Conclusion

Sprint 1.3 ajoute les visualisations de base pour enrichir les réponses avec des graphiques Chart.js.

**Bénéfices attendus**:
- ✅ Réponses visuellement enrichies
- ✅ KPI cards pour métriques clés
- ✅ Graphiques Chart.js prêts pour frontend
- ✅ Mapping automatique intent → visualisation

**Prochaine étape après v3.2.6.3**:
- Phase 2: Reasoning Agent (si roadmap le prévoit)
- OU Optimisations/améliorations Phase 1

**Baseline préservée**: v3.2.6.2 reste stable et rollback possible à tout moment ! ✅
