"""
Tests unitaires pour Analytics Agent

Couvre:
- Comparaisons temporelles (MoM, YoY)
- Détection anomalies (Z-score, IQR)
- Calcul tendances (régression linéaire)
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from conversation_service.agents.analytics.analytics_agent import (
    AnalyticsAgent,
    TimeSeriesMetrics,
    AnomalyDetectionResult,
    TrendAnalysis
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def analytics_agent():
    """Instance Analytics Agent avec config par défaut"""
    return AnalyticsAgent()


@pytest.fixture
def sample_transactions_january():
    """Transactions janvier 2025"""
    return [
        {'id': 1, 'amount': 120.50, 'date': '2025-01-05', 'merchant_name': 'Amazon'},
        {'id': 2, 'amount': 85.00, 'date': '2025-01-12', 'merchant_name': 'Carrefour'},
        {'id': 3, 'amount': 1200.00, 'date': '2025-01-15', 'merchant_name': 'Tesla'},  # Anomalie
        {'id': 4, 'amount': 95.75, 'date': '2025-01-20', 'merchant_name': 'Netflix'},
        {'id': 5, 'amount': 110.00, 'date': '2025-01-25', 'merchant_name': 'Uber'},
    ]


@pytest.fixture
def sample_transactions_december():
    """Transactions décembre 2024"""
    return [
        {'id': 10, 'amount': 110.00, 'date': '2024-12-05', 'merchant_name': 'Amazon'},
        {'id': 11, 'amount': 80.00, 'date': '2024-12-12', 'merchant_name': 'Carrefour'},
        {'id': 12, 'amount': 90.00, 'date': '2024-12-20', 'merchant_name': 'Netflix'},
    ]


# ============================================
# TESTS COMPARAISONS PÉRIODES
# ============================================

@pytest.mark.asyncio
async def test_compare_periods_sum(analytics_agent, sample_transactions_january, sample_transactions_december):
    """Test comparaison somme totale MoM"""
    result = await analytics_agent.compare_periods(
        transactions_current=sample_transactions_january,
        transactions_previous=sample_transactions_december,
        metric='sum'
    )

    # Validations
    assert isinstance(result, TimeSeriesMetrics)
    assert result.value_current == pytest.approx(1611.25, rel=0.01)  # Somme janvier
    assert result.value_previous == pytest.approx(280.00, rel=0.01)  # Somme décembre
    assert result.delta == pytest.approx(1331.25, rel=0.01)
    assert result.delta_percentage > 0  # Augmentation
    assert result.trend == 'up'


@pytest.mark.asyncio
async def test_compare_periods_avg(analytics_agent, sample_transactions_january, sample_transactions_december):
    """Test comparaison moyenne"""
    result = await analytics_agent.compare_periods(
        transactions_current=sample_transactions_january,
        transactions_previous=sample_transactions_december,
        metric='avg'
    )

    assert result.value_current == pytest.approx(322.25, rel=0.01)  # Moyenne janvier
    assert result.value_previous == pytest.approx(93.33, rel=0.1)   # Moyenne décembre


@pytest.mark.asyncio
async def test_compare_periods_stable(analytics_agent):
    """Test détection tendance stable (<5% variation)"""
    txs_current = [{'id': i, 'amount': 100, 'date': f'2025-01-{i+1:02d}'} for i in range(10)]
    txs_previous = [{'id': i+100, 'amount': 102, 'date': f'2024-12-{i+1:02d}'} for i in range(10)]

    result = await analytics_agent.compare_periods(
        transactions_current=txs_current,
        transactions_previous=txs_previous,
        metric='sum'
    )

    assert result.trend == 'stable'  # +2% < 5% threshold


@pytest.mark.asyncio
async def test_compare_periods_empty_raises_error(analytics_agent):
    """Test erreur si liste vide"""
    with pytest.raises(ValueError, match="cannot be empty"):
        await analytics_agent.compare_periods(
            transactions_current=[],
            transactions_previous=[{'id': 1, 'amount': 100, 'date': '2024-12-01'}],
            metric='sum'
        )


@pytest.mark.asyncio
async def test_compare_periods_invalid_metric_raises_error(analytics_agent, sample_transactions_january):
    """Test erreur si métrique invalide"""
    with pytest.raises(ValueError, match="Invalid metric"):
        await analytics_agent.compare_periods(
            transactions_current=sample_transactions_january,
            transactions_previous=sample_transactions_january,
            metric='invalid_metric'
        )


# ============================================
# TESTS DÉTECTION ANOMALIES
# ============================================

@pytest.mark.asyncio
async def test_detect_anomalies_zscore(analytics_agent, sample_transactions_january):
    """Test détection anomalies Z-score (Tesla 1200€ doit être détecté)"""
    anomalies = await analytics_agent.detect_anomalies(
        transactions=sample_transactions_january,
        method='zscore',
        threshold=1.5  # Seuil plus bas pour dataset de test
    )

    # Validations
    assert len(anomalies) >= 1

    # Transaction Tesla doit être anomalie
    tesla_anomaly = next((a for a in anomalies if a.merchant_name == 'Tesla'), None)
    assert tesla_anomaly is not None
    assert tesla_anomaly.amount == 1200.00
    assert tesla_anomaly.anomaly_score > 1.5
    assert 'σ de la moyenne' in tesla_anomaly.explanation


@pytest.mark.asyncio
async def test_detect_anomalies_iqr(analytics_agent, sample_transactions_january):
    """Test détection anomalies IQR"""
    anomalies = await analytics_agent.detect_anomalies(
        transactions=sample_transactions_january,
        method='iqr'
    )

    assert len(anomalies) >= 1
    assert all(a.method == 'iqr' for a in anomalies)
    assert all('IQR' in a.explanation for a in anomalies)


@pytest.mark.asyncio
async def test_detect_anomalies_empty_returns_empty(analytics_agent):
    """Test liste vide retourne liste vide (pas d'erreur)"""
    anomalies = await analytics_agent.detect_anomalies(
        transactions=[],
        method='zscore'
    )

    assert anomalies == []


@pytest.mark.asyncio
async def test_detect_anomalies_no_outliers(analytics_agent):
    """Test aucune anomalie si données uniformes"""
    uniform_txs = [{'id': i, 'amount': 100, 'date': f'2025-01-{i+1:02d}', 'merchant_name': 'Test'} for i in range(20)]

    anomalies = await analytics_agent.detect_anomalies(
        transactions=uniform_txs,
        method='zscore',
        threshold=2.0
    )

    assert len(anomalies) == 0


# ============================================
# TESTS CALCUL TENDANCES
# ============================================

@pytest.mark.asyncio
async def test_calculate_trend_increasing(analytics_agent):
    """Test tendance croissante"""
    # Données linéaires croissantes
    txs = [
        {'id': i, 'amount': 100 + i * 20, 'date': f'2024-{7+i:02d}-01'}
        for i in range(6)  # Juillet à décembre
    ]

    trend = await analytics_agent.calculate_trend(
        transactions=txs,
        aggregation='monthly',
        forecast_periods=3
    )

    # Validations
    assert trend.trend_direction == 'increasing'
    assert trend.slope > 0
    assert trend.r_squared > 0.95  # Excellente corrélation linéaire
    assert len(trend.forecast_next_periods) == 3
    assert all(trend.forecast_next_periods[i] < trend.forecast_next_periods[i+1] for i in range(2))  # Croissant


@pytest.mark.asyncio
async def test_calculate_trend_decreasing(analytics_agent):
    """Test tendance décroissante"""
    txs = [
        {'id': i, 'amount': 500 - i * 30, 'date': f'2024-{7+i:02d}-01'}
        for i in range(6)
    ]

    trend = await analytics_agent.calculate_trend(
        transactions=txs,
        aggregation='monthly',
        forecast_periods=3
    )

    assert trend.trend_direction == 'decreasing'
    assert trend.slope < 0


@pytest.mark.asyncio
async def test_calculate_trend_confidence_intervals(analytics_agent):
    """Test intervalles de confiance 95%"""
    # Données avec variabilité pour avoir intervalle confiance non-nul
    txs = [
        {'id': 1, 'amount': 100, 'date': '2024-07-01'},
        {'id': 2, 'amount': 115, 'date': '2024-08-01'},
        {'id': 3, 'amount': 125, 'date': '2024-09-01'},
        {'id': 4, 'amount': 138, 'date': '2024-10-01'},
        {'id': 5, 'amount': 155, 'date': '2024-11-01'},
        {'id': 6, 'amount': 162, 'date': '2024-12-01'},
    ]

    trend = await analytics_agent.calculate_trend(
        transactions=txs,
        aggregation='monthly',
        forecast_periods=3
    )

    # Validations intervalles
    assert len(trend.confidence_interval_95) == 3
    for i, ((lower, upper), forecast) in enumerate(zip(trend.confidence_interval_95, trend.forecast_next_periods)):
        # L'intervalle doit contenir le forecast (ou être très proche en cas de régression parfaite)
        tolerance = 0.01  # 1% tolérance
        assert lower <= forecast * (1 + tolerance), f"Period {i}: lower={lower} should be <= forecast={forecast}"
        assert upper >= forecast * (1 - tolerance), f"Period {i}: upper={upper} should be >= forecast={forecast}"


@pytest.mark.asyncio
async def test_calculate_trend_insufficient_data_raises_error(analytics_agent):
    """Test erreur si données insuffisantes (<3 points)"""
    txs = [
        {'id': 1, 'amount': 100, 'date': '2024-12-01'},
        {'id': 2, 'amount': 110, 'date': '2025-01-01'}
    ]

    with pytest.raises(ValueError, match="at least 3 data points"):
        await analytics_agent.calculate_trend(txs, 'monthly', 3)


# ============================================
# TESTS COVERAGE
# ============================================

def test_analytics_agent_initialization():
    """Test initialisation avec config custom"""
    agent = AnalyticsAgent(
        zscore_threshold=3.0,
        iqr_multiplier=2.0,
        stable_threshold_pct=10.0
    )

    assert agent.zscore_threshold == 3.0
    assert agent.iqr_multiplier == 2.0
    assert agent.stable_threshold_pct == 10.0


@pytest.mark.asyncio
async def test_period_label_extraction(analytics_agent):
    """Test extraction labels périodes"""
    txs = [
        {'id': 1, 'amount': 100, 'date': '2025-01-05'},
        {'id': 2, 'amount': 150, 'date': '2025-01-25'}
    ]

    result = await analytics_agent.compare_periods(
        transactions_current=txs,
        transactions_previous=txs,
        metric='sum'
    )

    assert '2025-01-05' in result.period_current
    assert '2025-01-25' in result.period_current
