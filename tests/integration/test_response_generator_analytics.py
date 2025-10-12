"""
Tests d'intégration Response Generator + Analytics Agent (Sprint 1.1)

Valide que l'Analytics Agent est correctement intégré dans le Response Generator
et que le fallback gracieux fonctionne si Analytics Agent échoue.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from conversation_service.agents.llm.response_generator import (
    ResponseGenerator,
    ResponseGenerationRequest,
    GeneratedInsight,
    InsightType
)
from conversation_service.agents.analytics.analytics_agent import (
    AnalyticsAgent,
    AnomalyDetectionResult
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def mock_llm_manager():
    """Mock LLM Manager pour éviter appels DeepSeek"""
    mock = Mock()
    mock.generate = AsyncMock()
    mock.generate_stream = AsyncMock()
    return mock


@pytest.fixture
def response_generator_with_analytics(mock_llm_manager):
    """Response Generator avec Analytics Agent activé"""
    generator = ResponseGenerator(
        llm_manager=mock_llm_manager,
        model="deepseek-chat",
        max_tokens=8000,
        temperature=0.7,
        enable_analytics=True  # Analytics Agent activé
    )
    return generator


@pytest.fixture
def response_generator_without_analytics(mock_llm_manager):
    """Response Generator avec Analytics Agent désactivé"""
    generator = ResponseGenerator(
        llm_manager=mock_llm_manager,
        model="deepseek-chat",
        max_tokens=8000,
        temperature=0.7,
        enable_analytics=False  # Analytics Agent désactivé
    )
    return generator


@pytest.fixture
def sample_transactions_with_anomaly():
    """Transactions réalistes avec une anomalie Tesla 1200€"""
    base_date = datetime(2025, 1, 1)
    return [
        {'id': 1, 'amount': -85.50, 'date': '2025-01-05', 'merchant_name': 'Carrefour', 'transaction_type': 'debit'},
        {'id': 2, 'amount': -42.30, 'date': '2025-01-06', 'merchant_name': 'Boulangerie', 'transaction_type': 'debit'},
        {'id': 3, 'amount': -1200.00, 'date': '2025-01-15', 'merchant_name': 'Tesla', 'transaction_type': 'debit'},  # Anomalie
        {'id': 4, 'amount': -95.75, 'date': '2025-01-20', 'merchant_name': 'Netflix', 'transaction_type': 'debit'},
        {'id': 5, 'amount': -110.00, 'date': '2025-01-25', 'merchant_name': 'Uber', 'transaction_type': 'debit'},
    ]


@pytest.fixture
def sample_request(sample_transactions_with_anomaly):
    """Requête de génération de réponse avec transactions"""
    return ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes dernières transactions",
        search_results=sample_transactions_with_anomaly,
        conversation_context=[],
        user_profile={"avg_monthly_spending": 1000.0},
        user_id=1,
        conversation_id="test-conv-123",
        generate_insights=True,
        search_aggregations=None
    )


# ============================================
# TESTS INTÉGRATION ANALYTICS AGENT
# ============================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_analytics_agent_initialization_enabled(response_generator_with_analytics):
    """Test que Analytics Agent est initialisé si enable_analytics=True"""

    # Validation
    assert response_generator_with_analytics.enable_analytics is True
    assert response_generator_with_analytics.analytics_agent is not None
    assert isinstance(response_generator_with_analytics.analytics_agent, AnalyticsAgent)

    # Statistiques initialisées
    assert "analytics_insights_generated" in response_generator_with_analytics.stats
    assert "analytics_fallbacks" in response_generator_with_analytics.stats
    assert response_generator_with_analytics.stats["analytics_insights_generated"] == 0
    assert response_generator_with_analytics.stats["analytics_fallbacks"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analytics_agent_initialization_disabled(response_generator_without_analytics):
    """Test que Analytics Agent n'est pas initialisé si enable_analytics=False"""

    # Validation
    assert response_generator_without_analytics.enable_analytics is False
    assert response_generator_without_analytics.analytics_agent is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_anomalies_with_analytics_agent_success(response_generator_with_analytics, sample_transactions_with_anomaly):
    """Test détection anomalie Tesla 1200€ avec Analytics Agent"""

    # Appeler la méthode directement
    insight = await response_generator_with_analytics._detect_anomalies_with_analytics_agent(
        search_results=sample_transactions_with_anomaly,
        user_profile={"avg_monthly_spending": 1000.0}
    )

    # Validations
    assert insight is not None
    assert isinstance(insight, GeneratedInsight)
    assert insight.type == InsightType.UNUSUAL_TRANSACTION
    assert "Analytics" in insight.title  # Titre contient "(Analytics)"
    assert "Tesla" in insight.description
    assert "1200" in insight.description
    assert insight.confidence > 0.7
    assert insight.data_support["analytics_agent_used"] is True
    assert insight.data_support["merchant_name"] == "Tesla"
    assert insight.data_support["anomaly_score"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_anomalies_insufficient_transactions(response_generator_with_analytics):
    """Test que Analytics Agent retourne None si <3 transactions"""

    # Seulement 2 transactions
    insufficient_txs = [
        {'id': 1, 'amount': -100.0, 'date': '2025-01-01', 'merchant_name': 'Test1', 'transaction_type': 'debit'},
        {'id': 2, 'amount': -200.0, 'date': '2025-01-02', 'merchant_name': 'Test2', 'transaction_type': 'debit'}
    ]

    insight = await response_generator_with_analytics._detect_anomalies_with_analytics_agent(
        search_results=insufficient_txs,
        user_profile={}
    )

    # Validation: None car <3 transactions
    assert insight is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_detect_anomalies_no_anomalies_detected(response_generator_with_analytics):
    """Test que Analytics Agent retourne None si aucune anomalie"""

    # Transactions uniformes (pas d'anomalie)
    uniform_txs = [
        {'id': i, 'amount': -100.0, 'date': f'2025-01-{i+1:02d}', 'merchant_name': f'Merchant{i}', 'transaction_type': 'debit'}
        for i in range(10)
    ]

    insight = await response_generator_with_analytics._detect_anomalies_with_analytics_agent(
        search_results=uniform_txs,
        user_profile={}
    )

    # Validation: None car pas d'anomalie significative
    assert insight is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analytics_agent_fallback_on_error(response_generator_with_analytics):
    """Test que le fallback se déclenche si Analytics Agent échoue"""

    # Mock Analytics Agent pour lever une exception
    original_agent = response_generator_with_analytics.analytics_agent
    response_generator_with_analytics.analytics_agent.detect_anomalies = AsyncMock(
        side_effect=Exception("Analytics Agent error simulation")
    )

    # Préparer transactions
    txs = [
        {'id': 1, 'amount': -100.0, 'date': '2025-01-01', 'merchant_name': 'Test1', 'transaction_type': 'debit'},
        {'id': 2, 'amount': -150.0, 'date': '2025-01-02', 'merchant_name': 'Test2', 'transaction_type': 'debit'},
        {'id': 3, 'amount': -500.0, 'date': '2025-01-03', 'merchant_name': 'Test3', 'transaction_type': 'debit'},
    ]

    # Appeler _generate_unusual_transaction_insight (qui gère le fallback)
    insight = await response_generator_with_analytics._generate_unusual_transaction_insight(
        search_results=txs,
        user_profile={}
    )

    # Validation: Fallback vers méthode basique
    # L'insight ne devrait PAS contenir "(Analytics)" car fallback utilisé
    if insight:
        assert "(Analytics)" not in insight.title
        assert insight.data_support.get("analytics_agent_used") is None

    # Statistiques fallback incrémentées
    assert response_generator_with_analytics.stats["analytics_fallbacks"] > 0

    # Restaurer Analytics Agent
    response_generator_with_analytics.analytics_agent = original_agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_insights_uses_analytics_agent(response_generator_with_analytics, sample_request):
    """Test que generate_automatic_insights utilise Analytics Agent"""

    # Générer insights automatiques
    insights = await response_generator_with_analytics._generate_automatic_insights(sample_request)

    # Validations
    assert len(insights) > 0

    # Vérifier qu'au moins un insight provient d'Analytics Agent
    analytics_insights = [
        insight for insight in insights
        if insight.data_support.get("analytics_agent_used") is True
    ]

    # Au moins 1 insight Analytics si anomalie détectée
    # (Tesla 1200€ devrait être détecté)
    assert len(analytics_insights) >= 1

    # Statistique analytics_insights_generated incrémentée
    # Note: Cette statistique est incrémentée dans _generate_unusual_transaction_insight
    # On ne peut pas la vérifier ici car elle dépend de l'appel complet


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analytics_statistics_tracking(response_generator_with_analytics, sample_transactions_with_anomaly):
    """Test que les statistiques Analytics sont correctement trackées"""

    # Stats initiales
    initial_analytics = response_generator_with_analytics.stats["analytics_insights_generated"]
    initial_fallbacks = response_generator_with_analytics.stats["analytics_fallbacks"]

    # Appeler _generate_unusual_transaction_insight (qui incrémente stats)
    insight = await response_generator_with_analytics._generate_unusual_transaction_insight(
        search_results=sample_transactions_with_anomaly,
        user_profile={}
    )

    # Validation insight généré
    assert insight is not None
    assert "(Analytics)" in insight.title

    # Statistiques incrémentées
    assert response_generator_with_analytics.stats["analytics_insights_generated"] == initial_analytics + 1
    assert response_generator_with_analytics.stats["analytics_fallbacks"] == initial_fallbacks  # Pas de fallback


# ============================================
# TESTS REGRESSION (Méthode basique préservée)
# ============================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_method_still_works_when_analytics_disabled(response_generator_without_analytics):
    """Test que la méthode basique fonctionne toujours si Analytics désactivé"""

    txs = [
        {'id': 1, 'amount': -100.0, 'date': '2025-01-01', 'merchant_name': 'Test1', 'transaction_type': 'debit'},
        {'id': 2, 'amount': -150.0, 'date': '2025-01-02', 'merchant_name': 'Test2', 'transaction_type': 'debit'},
        {'id': 3, 'amount': -500.0, 'date': '2025-01-03', 'merchant_name': 'Test3', 'transaction_type': 'debit'},
    ]

    # Appeler _generate_unusual_transaction_insight
    insight = await response_generator_without_analytics._generate_unusual_transaction_insight(
        search_results=txs,
        user_profile={}
    )

    # Validation: Méthode basique fonctionne toujours
    if insight:
        assert "(Analytics)" not in insight.title
        assert insight.type == InsightType.UNUSUAL_TRANSACTION


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_regression_v3_2_6_behavior(response_generator_without_analytics):
    """Test de non-régression : comportement v3.2.6 préservé"""

    # Scénario v3.2.6: Générer insights sans Analytics Agent
    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes transactions",
        search_results=[
            {'id': 1, 'amount': -100.0, 'date': '2025-01-01', 'merchant_name': 'Test', 'transaction_type': 'debit'},
        ],
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=True,
        search_aggregations=None
    )

    # Générer insights
    insights = await response_generator_without_analytics._generate_automatic_insights(request)

    # Validation: Pas de crash, comportement v3.2.6 préservé
    # (Peut générer 0 insights si pas assez de données, c'est normal)
    assert isinstance(insights, list)

    # Aucun insight Analytics généré
    analytics_insights = [
        i for i in insights
        if i.data_support.get("analytics_agent_used") is True
    ]
    assert len(analytics_insights) == 0
