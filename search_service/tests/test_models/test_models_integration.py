"""
🧪 Tests Intégration Modèles Search Service - Validation Complète
================================================================

Tests d'intégration pour valider que tous les modèles fonctionnent ensemble
selon l'architecture hybride Search Service.

Tests Coverage:
- Import et initialisation de tous les modèles
- Sérialisation/désérialisation Pydantic
- Validation des contrats interface
- Compatibilité entre modèles
- Tests bout-en-bout workflow
"""

import pytest
import json
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, List
from pydantic import ValidationError

# Imports des modèles à tester
from search_service.models import (
    # Contrats
    SearchServiceQuery, SearchServiceResponse, ContractValidator,
    QueryType, IntentType, FilterOperator, AggregationType,
    QueryMetadata, SearchParameters, FilterGroup, AggregationGroup,
    TransactionResult, ResponseMetadata, PerformanceMetrics,
    
    # Requêtes
    SimpleLexicalSearchRequest, CategorySearchRequest, RequestFactory,
    
    # Réponses  
    BaseResponse, SimpleLexicalSearchResponse, ResponseFactory,
    ResponseStatus, QueryComplexity,
    
    # Filtres
    CompositeFilter, FilterBuilder, AmountFilter, DateFilter,
    AmountFilterType, DateFilterType, TransactionType,
    
    # Elasticsearch
    ElasticsearchQuery, ESBoolQuery, ESQueryClause, ESQueryType,
    ElasticsearchQueryBuilder, FinancialQueryFactory,
    
    # Utilitaires
    validate_model_imports, list_available_models
)


# =============================================================================
# 🧪 TESTS IMPORTS ET INITIALISATION
# =============================================================================

class TestModelsImports:
    """Tests imports et disponibilité modèles."""
    
    def test_all_models_import_successfully(self):
        """Test que tous les modèles s'importent sans erreur."""
        validation_result = validate_model_imports()
        
        if isinstance(validation_result, tuple):
            success, error = validation_result
            assert success, f"Import validation failed: {error}"
        else:
            assert validation_result is True
    
    def test_available_models_listing(self):
        """Test listing modèles disponibles."""
        available_models = list_available_models()
        
        # Vérifier structure
        assert isinstance(available_models, dict)
        required_categories = {"contracts", "requests", "responses", "filters", "elasticsearch"}
        assert all(cat in available_models for cat in required_categories)
        
        # Vérifier contenu
        assert "SearchServiceQuery" in available_models["contracts"]
        assert "SimpleLexicalSearchRequest" in available_models["requests"]
        assert "BaseResponse" in available_models["responses"]
    
    def test_model_categories_not_empty(self):
        """Test que chaque catégorie contient des modèles."""
        available_models = list_available_models()
        
        for category, models in available_models.items():
            assert len(models) > 0, f"Category {category} is empty"
            
            # Vérifier que ce sont des strings
            for model_name in models:
                assert isinstance(model_name, str)
                assert len(model_name) > 0


# =============================================================================
# 🤝 TESTS CONTRATS INTERFACE
# =============================================================================

class TestServiceContracts:
    """Tests contrats interface Search Service."""
    
    def test_search_service_query_creation(self):
        """Test création SearchServiceQuery valide."""
        query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=34,
                intent_type=IntentType.SEARCH_BY_CATEGORY,
                confidence=0.95,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type=QueryType.FILTERED_SEARCH,
                fields=["searchable_text"],
                limit=20
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": 34},
                    {"field": "category_name", "operator": FilterOperator.EQ, "value": "restaurant"}
                ]
            )
        )
        
        assert query.query_metadata.user_id == 34
        assert query.query_metadata.intent_type == IntentType.SEARCH_BY_CATEGORY
        assert len(query.filters.required) == 2
    
    def test_search_service_query_validation(self):
        """Test validation SearchServiceQuery."""
        # Query sans user_id filter (devrait échouer)
        with pytest.raises(ValidationError, match="user_id filter is mandatory"):
            SearchServiceQuery(
                query_metadata=QueryMetadata(
                    user_id=34,
                    intent_type=IntentType.TEXT_SEARCH,
                    confidence=0.8,
                    agent_name="test_agent"
                ),
                search_parameters=SearchParameters(
                    query_type=QueryType.SIMPLE_SEARCH,
                    fields=["searchable_text"]
                ),
                filters=FilterGroup(
                    required=[
                        {"field": "category_name", "operator": FilterOperator.EQ, "value": "restaurant"}
                    ]
                )
            )
    
    def test_search_service_response_creation(self):
        """Test création SearchServiceResponse valide."""
        response = SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id="test-123",
                execution_time_ms=45,
                total_hits=1,  # ← Changé de 10 à 1
                returned_hits=1,  # ← Cohérent avec 1 résultat
                has_more=False,
                cache_hit=False,
                elasticsearch_took=30
            ),
            results=[
                TransactionResult(
                    transaction_id="tx_001",
                    user_id=34,
                    amount=-25.50,
                    amount_abs=25.50,
                    transaction_type="debit",
                    currency_code="EUR",
                    date="2024-01-15",
                    primary_description="RESTAURANT",
                    score=0.95
                )
            ],
            performance=PerformanceMetrics(
                query_complexity="simple",
                optimization_applied=["user_filter"],
                index_used="harena_transactions",
                shards_queried=1
            ),
            context_enrichment={
                "search_intent_matched": True,
                "result_quality_score": 0.95,
                "suggested_followup_questions": []
            }
        )
        
        assert response.response_metadata.total_hits == 1  # ← Changé de 10 à 1
        assert response.response_metadata.returned_hits == len(response.results)
        assert response.results[0].user_id == 34
    
    def test_contract_validator(self):
        """Test validateur contrats."""
        # Requête valide
        valid_query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=34,
                intent_type=IntentType.SEARCH_BY_CATEGORY,
                confidence=0.9,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type=QueryType.FILTERED_SEARCH,
                limit=20
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": 34}
                ]
            )
        )
        
        validation = ContractValidator.validate_search_query(valid_query)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_contract_serialization(self):
        """Test sérialisation JSON des contrats."""
        query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=34,
                intent_type=IntentType.TEXT_SEARCH,
                confidence=0.85,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type=QueryType.TEXT_SEARCH,
                fields=["searchable_text", "primary_description"]
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": 34}
                ]
            )
        )
        
        # Sérialisation
        json_str = query.json()
        assert isinstance(json_str, str)
        
        # Désérialisation
        query_dict = json.loads(json_str)
        restored_query = SearchServiceQuery.parse_obj(query_dict)
        
        assert restored_query.query_metadata.user_id == 34
        assert restored_query.query_metadata.intent_type == IntentType.TEXT_SEARCH


# =============================================================================
# 📥📤 TESTS MODÈLES REQUÊTES/RÉPONSES
# =============================================================================

class TestRequestResponseModels:
    """Tests modèles requêtes et réponses API."""
    
    def test_simple_lexical_search_request(self):
        """Test création requête recherche simple."""
        request = SimpleLexicalSearchRequest(
            query="restaurant italien",
            user_id=34,
            fields=["searchable_text", "primary_description"],
            limit=20
        )
        
        assert request.query == "restaurant italien"
        assert request.user_id == 34
        assert len(request.fields) == 2
    
    def test_category_search_request(self):
        """Test création requête recherche catégorie."""
        request = CategorySearchRequest(
            user_id=34,
            category="restaurant",
            date_from=date(2024, 1, 1),
            date_to=date(2024, 1, 31),
            include_stats=True
        )
        
        assert request.user_id == 34
        assert request.category == "restaurant"
        assert request.include_stats is True
    
    def test_request_factory(self):
        """Test factory création requêtes."""
        # Simple search
        simple_req = RequestFactory.create_simple_search(
            user_id=34,
            query="restaurant",
            limit=10
        )
        assert isinstance(simple_req, SimpleLexicalSearchRequest)
        assert simple_req.user_id == 34
        
        # Category search
        cat_req = RequestFactory.create_category_search(
            user_id=34,
            category="transport"
        )
        assert isinstance(cat_req, CategorySearchRequest)
        assert cat_req.category == "transport"
    
    def test_base_response_creation(self):
        """Test création réponse de base."""
        response = BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="Operation successful",
            execution_time_ms=150
        )
        
        assert response.status == ResponseStatus.SUCCESS
        assert response.execution_time_ms == 150
        assert isinstance(response.timestamp, datetime)
    
    def test_response_factory(self):
        """Test factory création réponses."""
        # Success response
        success_resp = ResponseFactory.create_success_response(
            message="Search completed",
            execution_time_ms=75
        )
        assert success_resp.status == ResponseStatus.SUCCESS
        assert success_resp.execution_time_ms == 75
        
        # Error response
        error_resp = ResponseFactory.create_error_response(
            error_code="VALIDATION_ERROR",
            error_message="Invalid parameter"
        )
        assert error_resp.status == ResponseStatus.ERROR
        assert error_resp.error_code == "VALIDATION_ERROR"


# =============================================================================
# 🔧 TESTS MODÈLES FILTRES
# =============================================================================

class TestFilterModels:
    """Tests modèles filtres spécialisés."""
    
    def test_amount_filter_creation(self):
        """Test création filtre montant."""
        amount_filter = AmountFilter(
            amount_type=AmountFilterType.ABSOLUTE,
            min_amount=Decimal("10.0"),
            max_amount=Decimal("100.0"),
            currency="EUR"
        )
        
        assert amount_filter.amount_type == AmountFilterType.ABSOLUTE
        assert amount_filter.min_amount == Decimal("10.0")
        assert amount_filter.max_amount == Decimal("100.0")
    
    def test_amount_filter_to_search_filters(self):
        """Test conversion filtre montant vers SearchFilters."""
        amount_filter = AmountFilter(
            amount_type=AmountFilterType.ABSOLUTE,
            min_amount=Decimal("50.0"),
            max_amount=Decimal("200.0")
        )
        
        search_filters = amount_filter.to_search_filters()
        assert len(search_filters) >= 2  # min et max
        
        # Vérifier présence filtres min/max
        min_filter = next((f for f in search_filters if f.operator == FilterOperator.GTE), None)
        max_filter = next((f for f in search_filters if f.operator == FilterOperator.LTE), None)
        
        assert min_filter is not None
        assert max_filter is not None
        assert min_filter.value == 50.0
        assert max_filter.value == 200.0
    
    def test_date_filter_creation(self):
        """Test création filtre date."""
        date_filter = DateFilter(
            filter_type=DateFilterType.DATE_RANGE,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        assert date_filter.filter_type == DateFilterType.DATE_RANGE
        assert date_filter.start_date == date(2024, 1, 1)
        assert date_filter.end_date == date(2024, 1, 31)
    
    def test_filter_builder(self):
        """Test builder filtres composés."""
        composite_filter = (FilterBuilder(user_id=34)
                           .with_amount_range(min_amount=10.0, max_amount=100.0)
                           .with_categories(["restaurant", "transport"])
                           .with_last_days(30)
                           .build())
        
        assert isinstance(composite_filter, CompositeFilter)
        assert composite_filter.user_isolation.user_id == 34
        assert composite_filter.amount_filter is not None
        assert composite_filter.category_filter is not None
        assert composite_filter.date_filter is not None
    
    def test_composite_filter_to_search_filters(self):
        """Test conversion filtre composé vers SearchFilters."""
        composite_filter = (FilterBuilder(user_id=34)
                           .with_categories(["restaurant"])
                           .build())
        
        filter_groups = composite_filter.to_search_filters()
        
        assert "required" in filter_groups
        assert "optional" in filter_groups
        assert "ranges" in filter_groups
        
        # Vérifier filtre user_id présent
        user_filters = [f for f in filter_groups["required"] if f.field == "user_id"]
        assert len(user_filters) == 1
        assert user_filters[0].value == 34


# =============================================================================
# 🔍 TESTS MODÈLES ELASTICSEARCH
# =============================================================================

class TestElasticsearchModels:
    """Tests modèles requêtes Elasticsearch."""
    
    def test_es_query_clause_creation(self):
        """Test création clause requête ES."""
        clause = ESQueryClause(
            query_type=ESQueryType.TERM,
            field="user_id",
            value=34
        )
        
        assert clause.query_type == ESQueryType.TERM
        assert clause.field == "user_id"
        assert clause.value == 34
    
    def test_es_query_clause_to_elasticsearch(self):
        """Test conversion clause vers Elasticsearch."""
        clause = ESQueryClause(
            query_type=ESQueryType.TERM,
            field="category_name.keyword",
            value="restaurant"
        )
        
        es_dict = clause.to_elasticsearch()
        expected = {
            "term": {
                "category_name.keyword": {
                    "value": "restaurant"
                }
            }
        }
        
        assert es_dict == expected
    
    def test_es_bool_query_creation(self):
        """Test création requête bool ES."""
        bool_query = ESBoolQuery(
            must=[
                ESQueryClause(query_type=ESQueryType.TERM, field="user_id", value=34)
            ],
            filter=[
                ESQueryClause(query_type=ESQueryType.TERM, field="category_name.keyword", value="restaurant")
            ]
        )
        
        assert len(bool_query.must) == 1
        assert len(bool_query.filter) == 1
    
    def test_elasticsearch_query_complete(self):
        """Test requête Elasticsearch complète."""
        query = ElasticsearchQuery(
            query=ESQueryClause(query_type=ESQueryType.MATCH_ALL),
            size=20,
            from_=0,
            sort=[{"date": {"order": "desc"}}]
        )
        
        es_dict = query.to_elasticsearch()
        
        assert "query" in es_dict
        assert "size" in es_dict
        assert "sort" in es_dict
        assert es_dict["size"] == 20
    
    def test_elasticsearch_query_builder(self):
        """Test builder requêtes Elasticsearch."""
        query = (ElasticsearchQueryBuilder()
                .add_filter(ESQueryClause(query_type=ESQueryType.TERM, field="user_id", value=34))
                .add_must(ESQueryClause(query_type=ESQueryType.MATCH, field="searchable_text", value="restaurant"))
                .set_size(20)
                .add_sort("date", "desc")
                .build())
        
        assert isinstance(query, ElasticsearchQuery)
        assert query.size == 20
        assert len(query.sort) == 1
    
    def test_financial_query_factory(self):
        """Test factory requêtes financières."""
        # User isolation filter
        user_filter = FinancialQueryFactory.create_user_isolation_filter(34)
        assert user_filter.field == "user_id"
        assert user_filter.value == 34
        
        # Text search query
        text_query = FinancialQueryFactory.create_text_search_query(
            "restaurant italien",
            ["searchable_text", "primary_description"]
        )
        assert text_query.query_type == ESQueryType.MULTI_MATCH
        assert text_query.value == "restaurant italien"


# =============================================================================
# 🔗 TESTS INTÉGRATION BOUT-EN-BOUT
# =============================================================================

class TestEndToEndIntegration:
    """Tests intégration bout-en-bout workflow complet."""
    
    def test_complete_search_workflow(self):
        """Test workflow complet recherche."""
        # 1. Créer requête API
        api_request = SimpleLexicalSearchRequest(
            query="restaurant italien",
            user_id=34,
            limit=10
        )
        
        # 2. Convertir vers contrat
        contract_query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=api_request.user_id,
                intent_type=IntentType.TEXT_SEARCH,
                confidence=0.9,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type=QueryType.TEXT_SEARCH,
                fields=api_request.fields,
                limit=api_request.limit
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": api_request.user_id}
                ],
                text_search={
                    "query": api_request.query,
                    "fields": api_request.fields,
                    "operator": "match"
                }
            )
        )
        
        # 3. Valider contrat
        validation = ContractValidator.validate_search_query(contract_query)
        assert validation["valid"] is True
        
        # 4. Convertir vers Elasticsearch
        es_query = (ElasticsearchQueryBuilder()
                   .add_filter(FinancialQueryFactory.create_user_isolation_filter(34))
                   .add_must(FinancialQueryFactory.create_text_search_query(
                       "restaurant italien",
                       ["searchable_text", "primary_description"]
                   ))
                   .set_size(10)
                   .build())
        
        # 5. Valider requête ES
        es_validation = es_query.validate_query()
        assert es_validation["valid"] is True
        
        # 6. Créer réponse simulée
        mock_results = [
            TransactionResult(
                transaction_id="tx_001",
                user_id=34,
                amount=-45.67,
                amount_abs=45.67,
                transaction_type="debit",
                currency_code="EUR",
                date="2024-01-15",
                primary_description="RISTORANTE ITALIANO",
                score=0.95
            )
        ]
        
        api_response = ResponseFactory.create_search_response(
            results=mock_results,
            total_found=1,
            execution_time_ms=50,
            elasticsearch_time_ms=30
        )
        
        assert api_response.status == ResponseStatus.SUCCESS
        assert len(api_response.results) == 1
        assert api_response.results[0].user_id == 34
    
    def test_filter_to_elasticsearch_conversion(self):
        """Test conversion filtres vers Elasticsearch."""
        # Créer filtre composé
        composite_filter = (FilterBuilder(user_id=34)
                           .with_amount_range(min_amount=10.0, max_amount=100.0)
                           .with_categories(["restaurant"])
                           .build())
        
        # Convertir vers groupes de filtres
        filter_groups = composite_filter.to_search_filters()
        
        # Appliquer à builder Elasticsearch
        es_builder = ElasticsearchQueryBuilder()
        es_builder.apply_composite_filter(composite_filter)
        
        # Construire requête
        es_query = es_builder.set_size(20).build()
        
        # Vérifier structure ES
        es_dict = es_query.to_elasticsearch()
        assert "query" in es_dict
        assert "bool" in es_dict["query"]
        assert "filter" in es_dict["query"]["bool"]
        
        # Vérifier présence filtres
        filters = es_dict["query"]["bool"]["filter"]
        assert len(filters) > 0
        
        # Vérifier filtre user_id
        user_filter = next((f for f in filters if "term" in f and "user_id" in f["term"]), None)
        assert user_filter is not None
    
    def test_json_serialization_roundtrip(self):
        """Test sérialisation JSON complète bout-en-bout."""
        # Créer objets complexes
        original_query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=34,
                intent_type=IntentType.SEARCH_BY_CATEGORY,
                confidence=0.95,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type=QueryType.FILTERED_SEARCH,
                fields=["searchable_text", "primary_description"],
                limit=20
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": 34},
                    {"field": "category_name", "operator": FilterOperator.EQ, "value": "restaurant"}
                ]
            )
        )
        
        # Sérialisation JSON
        json_str = original_query.json()
        
        # Désérialisation
        query_dict = json.loads(json_str)
        restored_query = SearchServiceQuery.parse_obj(query_dict)
        
        # Vérification égalité
        assert restored_query.query_metadata.user_id == original_query.query_metadata.user_id
        assert restored_query.query_metadata.intent_type == original_query.query_metadata.intent_type
        assert len(restored_query.filters.required) == len(original_query.filters.required)


# =============================================================================
# 🔧 TESTS UTILITAIRES ET EDGE CASES
# =============================================================================

class TestModelsEdgeCases:
    """Tests cas limites et validation erreurs."""
    
    def test_invalid_user_id_validation(self):
        """Test validation user_id invalide."""
        with pytest.raises(ValidationError):
            QueryMetadata(
                user_id=0,  # Invalide
                intent_type=IntentType.TEXT_SEARCH,
                confidence=0.8,
                agent_name="test_agent"
            )
    
    def test_invalid_confidence_validation(self):
        """Test validation confidence invalide."""
        with pytest.raises(ValidationError):
            QueryMetadata(
                user_id=34,
                intent_type=IntentType.TEXT_SEARCH,
                confidence=1.5,  # > 1.0
                agent_name="test_agent"
            )
    
    def test_empty_query_validation(self):
        """Test validation requête vide."""
        with pytest.raises(ValidationError):
            SimpleLexicalSearchRequest(
                query="",  # Vide
                user_id=34
            )
    
    def test_large_limit_validation(self):
        """Test validation limite trop grande."""
        with pytest.raises(ValidationError):
            SearchParameters(
                query_type=QueryType.SIMPLE_SEARCH,
                limit=10000  # Trop grand
            )
    
    def test_amount_consistency_validation(self):
        """Test validation cohérence montants."""
        with pytest.raises(ValidationError):
            TransactionResult(
                transaction_id="tx_001",
                user_id=34,
                amount=-45.67,
                amount_abs=50.00,  # Incohérent avec amount
                transaction_type="debit",
                currency_code="EUR",
                date="2024-01-15",
                primary_description="TEST",
                score=0.95
            )
    
    def test_date_range_validation(self):
        """Test validation plage dates."""
        with pytest.raises(ValidationError):
            DateFilter(
                filter_type=DateFilterType.DATE_RANGE,
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1)  # Fin avant début
            )


# =============================================================================
# 📊 TESTS PERFORMANCE ET COMPLEXITÉ
# =============================================================================

class TestModelsPerformance:
    """Tests performance et complexité modèles."""
    
    def test_large_filter_list_handling(self):
        """Test gestion listes filtres importantes."""
        # Créer beaucoup de filtres
        many_categories = [f"category_{i}" for i in range(50)]
        
        # Ne devrait pas lever d'exception
        filter_builder = FilterBuilder(user_id=34).with_categories(many_categories)
        composite_filter = filter_builder.build()
        
        filter_groups = composite_filter.to_search_filters()
        assert len(filter_groups["required"]) > 0
    
    def test_complex_elasticsearch_query_building(self):
        """Test construction requête ES complexe."""
        # Builder avec beaucoup d'éléments
        complex_query = (ElasticsearchQueryBuilder()
                        .add_filter(FinancialQueryFactory.create_user_isolation_filter(34))
                        .add_filter(FinancialQueryFactory.create_category_filter(["restaurant", "transport"]))
                        .add_filter(FinancialQueryFactory.create_amount_range_filter(10.0, 1000.0))
                        .add_must(FinancialQueryFactory.create_text_search_query("test", ["searchable_text"]))
                        .set_size(100)
                        .add_sort("date", "desc")
                        .add_sort("amount_abs", "desc")
                        .build())
        
        # Validation
        validation = complex_query.validate_query()
        assert validation["valid"] is True
        
        # Conversion ES
        es_dict = complex_query.to_elasticsearch()
        assert isinstance(es_dict, dict)
        assert "query" in es_dict
    
    def test_serialization_performance(self):
        """Test performance sérialisation."""
        # Créer objet complexe
        large_response = SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id="test-large",
                execution_time_ms=100,
                total_hits=1000,
                returned_hits=100,
                has_more=True,
                cache_hit=False,
                elasticsearch_took=80
            ),
            results=[
                TransactionResult(
                    transaction_id=f"tx_{i:06d}",
                    user_id=34,
                    amount=float(-(i * 10.5)),
                    amount_abs=float(i * 10.5),
                    transaction_type="debit",
                    currency_code="EUR",
                    date="2024-01-15",
                    primary_description=f"TRANSACTION {i}",
                    score=1.0 - (i * 0.01)
                )
                for i in range(100)
            ],
            performance=PerformanceMetrics(
                query_complexity="complex",
                optimization_applied=["user_filter", "category_filter"],
                index_used="harena_transactions",
                shards_queried=3
            ),
            context_enrichment={
                "search_intent_matched": True,
                "result_quality_score": 0.85,
                "suggested_followup_questions": []
            }
        )
        
        # Sérialisation ne devrait pas être trop lente
        import time
        start_time = time.time()
        json_str = large_response.json()
        end_time = time.time()
        
        serialization_time = end_time - start_time
        assert serialization_time < 1.0  # Moins d'1 seconde
        assert len(json_str) > 1000  # Contenu substantiel


if __name__ == "__main__":
    """Exécution tests si script lancé directement."""
    pytest.main([__file__, "-v", "--tb=short"])