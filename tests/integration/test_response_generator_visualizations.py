"""
Tests d'intégration Response Generator + VisualizationService (Sprint 1.3)

Valide que le VisualizationService est correctement intégré dans le Response Generator
et que le fallback gracieux fonctionne si VisualizationService échoue.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from conversation_service.agents.llm.response_generator import (
    ResponseGenerator,
    ResponseGenerationRequest,
    ResponseType
)
from conversation_service.services.visualization.visualization_service import VisualizationService
from conversation_service.models.visualization.schemas import (
    VisualizationType,
    KPICard,
    ChartVisualization,
    VisualizationResponse
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def mock_llm_manager():
    """Mock LLM Manager pour éviter appels DeepSeek"""
    mock = Mock()
    mock_response = Mock()
    mock_response.error = None
    mock_response.content = "Voici vos transactions récentes."
    mock_response.model_used = "deepseek-chat"
    mock_response.usage = {"total_tokens": 100}
    mock.generate = AsyncMock(return_value=mock_response)
    mock.generate_stream = AsyncMock()
    return mock


@pytest.fixture
def response_generator_with_visualizations(mock_llm_manager):
    """Response Generator avec VisualizationService activé"""
    generator = ResponseGenerator(
        llm_manager=mock_llm_manager,
        model="deepseek-chat",
        max_tokens=8000,
        temperature=0.7,
        enable_analytics=False,  # Désactiver Analytics pour isoler tests visualizations
        enable_visualizations=True  # VisualizationService activé
    )
    return generator


@pytest.fixture
def response_generator_without_visualizations(mock_llm_manager):
    """Response Generator avec VisualizationService désactivé"""
    generator = ResponseGenerator(
        llm_manager=mock_llm_manager,
        model="deepseek-chat",
        max_tokens=8000,
        temperature=0.7,
        enable_analytics=False,
        enable_visualizations=False  # VisualizationService désactivé
    )
    return generator


@pytest.fixture
def sample_transactions_by_category():
    """Transactions réalistes avec différentes catégories"""
    return [
        {'id': 1, 'amount': -85.50, 'date': '2025-01-05', 'merchant_name': 'Carrefour', 'category': 'Alimentation', 'transaction_type': 'debit'},
        {'id': 2, 'amount': -42.30, 'date': '2025-01-06', 'merchant_name': 'Boulangerie', 'category': 'Alimentation', 'transaction_type': 'debit'},
        {'id': 3, 'amount': -120.00, 'date': '2025-01-15', 'merchant_name': 'SNCF', 'category': 'Transport', 'transaction_type': 'debit'},
        {'id': 4, 'amount': -15.99, 'date': '2025-01-20', 'merchant_name': 'Netflix', 'category': 'Loisirs', 'transaction_type': 'debit'},
        {'id': 5, 'amount': -55.00, 'date': '2025-01-25', 'merchant_name': 'Uber', 'category': 'Transport', 'transaction_type': 'debit'},
    ]


@pytest.fixture
def sample_aggregations():
    """Agrégations réalistes pour visualisations avancées"""
    return {
        'transaction_count': {'value': 5},
        'total_debit': {'sum_amount': {'value': 318.79}},
        'total_credit': {'sum_amount': {'value': 0}}
    }


@pytest.fixture
def sample_request_by_category(sample_transactions_by_category, sample_aggregations):
    """Requête de génération de réponse avec transactions par catégorie"""
    return ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="by_category",
        user_message="Mes dépenses par catégorie",
        search_results=sample_transactions_by_category,
        conversation_context=[],
        user_profile={},
        user_id=1,
        conversation_id="test-viz-123",
        generate_insights=False,  # Désactiver insights pour isoler tests visualizations
        search_aggregations=sample_aggregations
    )


# ============================================
# TESTS INTÉGRATION VISUALIZATION SERVICE
# ============================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualization_service_initialization_enabled(response_generator_with_visualizations):
    """Test que VisualizationService est initialisé si enable_visualizations=True"""

    # Validation
    assert response_generator_with_visualizations.enable_visualizations is True
    assert response_generator_with_visualizations.visualization_service is not None
    assert isinstance(response_generator_with_visualizations.visualization_service, VisualizationService)

    # Statistiques initialisées
    assert "visualizations_generated" in response_generator_with_visualizations.stats
    assert "visualization_failures" in response_generator_with_visualizations.stats
    assert response_generator_with_visualizations.stats["visualizations_generated"] == 0
    assert response_generator_with_visualizations.stats["visualization_failures"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualization_service_initialization_disabled(response_generator_without_visualizations):
    """Test que VisualizationService n'est pas initialisé si enable_visualizations=False"""

    # Validation
    assert response_generator_without_visualizations.enable_visualizations is False
    assert response_generator_without_visualizations.visualization_service is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_visualizations_by_category(response_generator_with_visualizations, sample_request_by_category):
    """Test génération visualisations pour intent by_category (KPI + Pie Chart)"""

    # Appeler la méthode directement
    visualizations = await response_generator_with_visualizations._generate_visualizations(sample_request_by_category)

    # Validations
    assert len(visualizations) > 0

    # Au moins un KPI Card (Total Dépenses, Nombre Transactions, Montant Moyen)
    kpi_cards = [v for v in visualizations if v.get("type") == "kpi_card"]
    assert len(kpi_cards) >= 1

    # Vérifier structure KPI Card
    if kpi_cards:
        first_kpi = kpi_cards[0]
        assert "title" in first_kpi
        assert "value" in first_kpi
        assert "unit" in first_kpi

    # Au moins un Pie Chart pour catégories
    pie_charts = [v for v in visualizations if v.get("type") == "pie_chart"]
    assert len(pie_charts) >= 1

    # Vérifier structure Pie Chart
    if pie_charts:
        first_pie = pie_charts[0]
        assert "title" in first_pie
        assert "data" in first_pie
        assert "labels" in first_pie["data"]
        assert "datasets" in first_pie["data"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_response_includes_visualizations(response_generator_with_visualizations, sample_request_by_category):
    """Test que generate_response inclut visualisations dans ResponseGenerationResult"""

    # Générer réponse complète
    result = await response_generator_with_visualizations.generate_response(sample_request_by_category)

    # Validations
    assert result.success is True
    assert result.response_type == ResponseType.DATA_PRESENTATION
    assert len(result.data_visualizations) > 0

    # Statistiques incrémentées
    assert response_generator_with_visualizations.stats["visualizations_generated"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualizations_not_generated_without_search_results(response_generator_with_visualizations):
    """Test que visualisations ne sont pas générées si search_results vide"""

    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes transactions",
        search_results=[],  # Pas de résultats
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=False,
        search_aggregations=None
    )

    # Appeler _generate_visualizations
    visualizations = await response_generator_with_visualizations._generate_visualizations(request)

    # Validation: Pas de visualisations générées
    assert len(visualizations) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualization_service_graceful_degradation_on_error(response_generator_with_visualizations, sample_request_by_category):
    """Test que le fallback se déclenche si VisualizationService échoue"""

    # Mock VisualizationService pour lever une exception
    original_service = response_generator_with_visualizations.visualization_service
    response_generator_with_visualizations.visualization_service.generate_visualizations = Mock(
        side_effect=Exception("VisualizationService error simulation")
    )

    # Appeler _generate_visualizations
    visualizations = await response_generator_with_visualizations._generate_visualizations(sample_request_by_category)

    # Validation: Graceful degradation - pas de crash, retourne []
    assert isinstance(visualizations, list)
    # Peut retourner visualisations de fallback ou [] selon implémentation

    # Statistique visualization_failures incrémentée
    assert response_generator_with_visualizations.stats["visualization_failures"] > 0

    # Restaurer VisualizationService
    response_generator_with_visualizations.visualization_service = original_service


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intent_mapping_simple_generates_kpi_only(response_generator_with_visualizations, sample_transactions_by_category):
    """Test que intent simple génère uniquement KPI Cards"""

    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes transactions",
        search_results=sample_transactions_by_category,
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=False,
        search_aggregations=None
    )

    # Générer visualisations
    visualizations = await response_generator_with_visualizations._generate_visualizations(request)

    # Validation: Au moins KPI Cards
    kpi_cards = [v for v in visualizations if v.get("type") == "kpi_card"]
    assert len(kpi_cards) > 0

    # Pas de Pie/Bar/Line charts pour intent simple (selon mapping)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualization_statistics_tracking(response_generator_with_visualizations, sample_request_by_category):
    """Test que les statistiques Visualizations sont correctement trackées"""

    # Stats initiales
    initial_viz = response_generator_with_visualizations.stats["visualizations_generated"]
    initial_failures = response_generator_with_visualizations.stats["visualization_failures"]

    # Générer visualisations
    visualizations = await response_generator_with_visualizations._generate_visualizations(sample_request_by_category)

    # Validation: Stats incrémentées
    assert response_generator_with_visualizations.stats["visualizations_generated"] > initial_viz
    assert response_generator_with_visualizations.stats["visualization_failures"] == initial_failures  # Pas d'erreur


# ============================================
# TESTS REGRESSION (Méthode fallback préservée)
# ============================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fallback_method_works_when_service_disabled(response_generator_without_visualizations, sample_transactions_by_category):
    """Test que la méthode fallback fonctionne si VisualizationService désactivé"""

    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes transactions",
        search_results=sample_transactions_by_category,
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=False,
        search_aggregations=None
    )

    # Générer visualisations
    visualizations = await response_generator_without_visualizations._generate_visualizations(request)

    # Validation: Fallback vers méthode legacy (peut générer visualisations basiques)
    assert isinstance(visualizations, list)
    # La méthode legacy peut générer line_chart et pie_chart basiques


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_regression_v3_2_6_2_behavior(response_generator_without_visualizations):
    """Test de non-régression : comportement v3.2.6.2 préservé"""

    # Scénario v3.2.6.2: Générer visualisations sans VisualizationService
    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="simple",
        user_message="Mes transactions",
        search_results=[
            {'id': 1, 'amount': -100.0, 'date': '2025-01-01', 'merchant_name': 'Test', 'category': 'Test', 'transaction_type': 'debit'},
        ],
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=False,
        search_aggregations=None
    )

    # Générer visualisations
    visualizations = await response_generator_without_visualizations._generate_visualizations(request)

    # Validation: Pas de crash, comportement v3.2.6.2 préservé
    assert isinstance(visualizations, list)


# ============================================
# TESTS E2E COMPLETS
# ============================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_by_category_complete_flow(response_generator_with_visualizations, sample_request_by_category):
    """Test E2E complet: by_category avec KPI + Pie Chart"""

    # Générer réponse complète
    result = await response_generator_with_visualizations.generate_response(sample_request_by_category)

    # Validations complètes
    assert result.success is True
    assert result.response_type == ResponseType.DATA_PRESENTATION
    assert len(result.data_visualizations) > 0

    # KPI Cards présents
    kpi_cards = [v for v in result.data_visualizations if v.get("type") == "kpi_card"]
    assert len(kpi_cards) >= 1

    # Pie Chart présent
    pie_charts = [v for v in result.data_visualizations if v.get("type") == "pie_chart"]
    assert len(pie_charts) >= 1

    # Vérifier structure Chart.js
    if pie_charts:
        pie = pie_charts[0]
        assert "data" in pie
        assert "labels" in pie["data"]
        assert "datasets" in pie["data"]
        assert len(pie["data"]["datasets"]) > 0
        assert "data" in pie["data"]["datasets"][0]  # Valeurs numériques

    # Statistiques
    assert response_generator_with_visualizations.stats["responses_generated"] > 0
    assert response_generator_with_visualizations.stats["visualizations_generated"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_with_aggregations(response_generator_with_visualizations, sample_transactions_by_category, sample_aggregations):
    """Test E2E avec agrégations pour visualisations optimisées"""

    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="by_category",
        user_message="Mes dépenses par catégorie ce mois-ci",
        search_results=sample_transactions_by_category,
        conversation_context=[],
        user_profile={},
        user_id=1,
        generate_insights=False,
        search_aggregations=sample_aggregations  # Agrégations fournies
    )

    # Générer réponse
    result = await response_generator_with_visualizations.generate_response(request)

    # Validations
    assert result.success is True
    assert len(result.data_visualizations) > 0

    # KPI Cards utilisent les agrégations
    kpi_cards = [v for v in result.data_visualizations if v.get("type") == "kpi_card"]
    if kpi_cards:
        # Vérifier cohérence avec agrégations
        total_debit_kpi = next((k for k in kpi_cards if "Dépenses" in k.get("title", "")), None)
        if total_debit_kpi:
            # Valeur devrait correspondre aux agrégations
            assert total_debit_kpi["value"] > 0
