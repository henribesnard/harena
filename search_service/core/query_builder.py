import logging
from typing import Dict, Any, List
from search_service.models.request import SearchRequest

logger = logging.getLogger(__name__)

class QueryBuilder:
    """Constructeur de requêtes Elasticsearch simple et efficace"""
    
    def __init__(self):
        # Configuration des champs de recherche basée sur votre architecture
        self.search_fields = [
            "searchable_text^2.0",      # Champ principal enrichi
            "primary_description^1.5",   # Description transaction
            "merchant_name^1.8",         # Nom marchand
            "category_name^1.0"          # Catégorie
        ]
    
    def build_query(self, request: SearchRequest) -> Dict[str, Any]:
        """
        Construction intelligente de requête Elasticsearch
        Gère automatiquement les différents cas d'usage
        """
        
        # Filtre obligatoire sur user_id pour sécurité
        must_filters = [
            {"term": {"user_id": request.user_id}}
        ]
        
        # Requête textuelle si fournie
        if request.query and request.query.strip():
            text_query = self._build_text_query(request.query)
            must_filters.append(text_query)
            logger.debug(f"Added text query for: '{request.query}'")
        
        # Filtres additionnels
        additional_filters = self._build_additional_filters(request.filters)
        must_filters.extend(additional_filters)
        
        # Construction requête finale
        query = {
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "sort": self._build_sort_criteria(request),
            "_source": self._get_source_fields()
        }
        
        logger.debug(f"Built query with {len(must_filters)} filters")
        return query
    
    def _build_text_query(self, query_text: str) -> Dict[str, Any]:
        """Construction de la requête textuelle optimisée"""
        terms_count = len(query_text.split())
        minimum_should_match = "50%" if terms_count > 2 else "100%"
        return {
            "multi_match": {
                "query": query_text,
                "fields": self.search_fields,
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": minimum_should_match
            }
        }
    
    def _build_additional_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construction des filtres additionnels"""
        filter_list = []

        for field, value in filters.items():
            if value is None:
                continue
            if isinstance(value, dict):
                # Filtre range (ex: {"amount": {"gte": -100, "lte": 0}})
                filter_list.append({"range": {field: value}})
                logger.debug(f"Added range filter on {field}: {value}")
                
            elif isinstance(value, list):
                # Filtre terms (ex: {"category_name": ["restaurant", "bar"]})
                filter_list.append({"terms": {field: value}})
                logger.debug(f"Added terms filter on {field}: {len(value)} values")
                
            else:
                # Filtre exact (ex: {"category_name": "restaurant"})
                # Utiliser .keyword pour les champs textuels si approprié
                field_name = self._get_filter_field_name(field)
                filter_list.append({"term": {field_name: value}})
                logger.debug(f"Added term filter on {field_name}: {value}")
        
        return filter_list
    
    def _get_filter_field_name(self, field: str) -> str:
        """
        Détermine le nom de champ correct pour les filtres
        Ajoute .keyword pour les champs textuels si nécessaire
        """
        # Champs qui nécessitent .keyword pour les filtres exacts
        keyword_fields = {
            'category_name', 'merchant_name', 'operation_type',
            'currency_code', 'transaction_type', 'weekday'
        }
        
        if field in keyword_fields:
            return f"{field}.keyword"
        else:
            return field
    
    def _build_sort_criteria(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """Construction des critères de tri"""
        sort_criteria = []
        
        # Si c'est une recherche textuelle, trier par score d'abord
        if request.query and request.query.strip():
            sort_criteria.append({"_score": {"order": "desc"}})
        
        # Toujours trier par date décroissante en second
        sort_criteria.append({"date": {"order": "desc"}})
        
        return sort_criteria
    
    def _get_source_fields(self) -> List[str]:
        """Définit les champs à retourner dans les résultats"""
        return [
            "transaction_id", "user_id", "account_id",
            "amount", "amount_abs", "currency_code", "transaction_type",
            "date", "month_year", "weekday",
            "primary_description", "merchant_name", "category_name", "operation_type"
        ]
    
    def build_aggregation_query(self, request: SearchRequest, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """Construction d'une requête avec agrégations simples"""
        base_query = self.build_query(request)

        aggregations: Dict[str, Any] = {}
        group_by = (aggregation or {}).get("group_by", [])
        metrics = (aggregation or {}).get("metrics", [])

        if group_by:
            for field in group_by:
                field_name = self._get_filter_field_name(field)
                agg_def: Dict[str, Any] = {
                    "terms": {"field": field_name, "size": 10}
                }
                sub_aggs: Dict[str, Any] = {}
                if "sum" in metrics:
                    sub_aggs["amount_sum"] = {"sum": {"field": "amount"}}
                if sub_aggs:
                    agg_def["aggs"] = sub_aggs
                aggregations[f"{field}_terms"] = agg_def
        elif "sum" in metrics:
            aggregations["amount_sum"] = {"sum": {"field": "amount"}}

        if aggregations:
            base_query["aggs"] = aggregations
            base_query["size"] = min(request.limit, 10)

        return base_query
