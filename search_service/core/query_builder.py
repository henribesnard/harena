import logging
from typing import Dict, Any, List, Optional
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
            "_source": self._get_source_fields(),
            "size": request.limit,
            "from": request.offset
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
    
    def build_aggregation_query(
        self, request: SearchRequest, aggregation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construction d'une requête avec agrégations Elasticsearch natives.
        
        ✅ CORRECTION MAJEURE : Supporte maintenant les deux formats :
        1. Format Elasticsearch natif (PRIORITÉ) : {"my_agg": {"sum": {"field": "amount"}}}
        2. Format abstrait legacy : {"group_by": [...], "metrics": [...]}
        
        Args:
            request: Requête de recherche de base.
            aggregation: Agrégations au format Elasticsearch natif ou abstrait.
        """
        base_query = self.build_query(request)

        if not aggregation:
            logger.debug("No aggregations provided")
            return base_query

        aggregations: Dict[str, Any] = {}

        # ✅ NOUVEAU : Détection et traitement du format Elasticsearch natif
        if self._is_elasticsearch_native_format(aggregation):
            logger.debug("Processing Elasticsearch native aggregation format")
            # Validation et passage direct des agrégations
            validated_aggs = self._validate_elasticsearch_aggregations(aggregation)
            aggregations = validated_aggs
        
        # ✅ ANCIEN : Support format abstrait pour compatibilité arrière
        else:
            logger.debug("Processing legacy abstract aggregation format")
            group_by = aggregation.get("group_by") or []
            metrics = aggregation.get("metrics") or []

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

        # Ajouter les agrégations à la requête finale
        if aggregations:
            base_query["aggs"] = aggregations
            logger.info(
                f"Added {len(aggregations)} aggregations to query: {list(aggregations.keys())}"
            )
            if getattr(request, "aggregation_only", False):
                base_query["size"] = 0
                logger.info(
                    "Aggregation-only request detected: returning only aggregations"
                )
        else:
            logger.warning("No valid aggregations generated from input")

        return base_query

    def _is_elasticsearch_native_format(self, aggregation: Dict[str, Any]) -> bool:
        """Détecte si les agrégations sont au format Elasticsearch natif.
        
        Format natif : {"total_sum": {"sum": {"field": "amount_abs"}}}
        Format abstrait : {"group_by": ["category"], "metrics": ["sum"]}
        """
        # Si contient les clés abstraites, c'est l'ancien format
        abstract_keys = {"group_by", "metrics", "types"}
        if any(key in aggregation for key in abstract_keys):
            return False
        
        # Vérifier que c'est bien du format Elasticsearch natif
        elasticsearch_agg_types = {
            # Métriques simples
            "sum", "avg", "min", "max", "value_count", "cardinality",
            "stats", "extended_stats", "percentiles", "percentile_ranks",
            
            # Buckets
            "terms", "date_histogram", "histogram", "range", "date_range",
            "filters", "filter", "missing", "nested", "reverse_nested",
            "global", "sampler", "diversified_sampler"
        }
        
        # Vérifier chaque agrégation
        for agg_name, agg_def in aggregation.items():
            if not isinstance(agg_def, dict):
                continue
                
            # Chercher un type d'agrégation ES valide au niveau racine
            agg_types_found = set(agg_def.keys()) & elasticsearch_agg_types
            if agg_types_found:
                return True
                
            # Vérifier les sous-agrégations (aggs)
            if "aggs" in agg_def and isinstance(agg_def["aggs"], dict):
                return True
        
        return False

    def _validate_elasticsearch_aggregations(self, aggregation: Dict[str, Any]) -> Dict[str, Any]:
        """Valide et nettoie les agrégations Elasticsearch natives.
        
        Effectue une validation basique pour éviter les injections malveillantes
        et s'assurer que les champs référencés existent.
        """
        # Champs autorisés pour les agrégations
        allowed_fields = {
            "amount", "amount_abs", "date", "transaction_id", "user_id", "account_id",
            "currency_code", "transaction_type", "operation_type", "category_name",
            "merchant_name", "primary_description", "month_year", "weekday"
        }
        
        validated = {}
        
        for agg_name, agg_def in aggregation.items():
            # Validation nom agrégation (pas d'injection)
            if not isinstance(agg_name, str) or len(agg_name) > 100:
                logger.warning(f"Invalid aggregation name: {agg_name}")
                continue
                
            if not isinstance(agg_def, dict):
                logger.warning(f"Invalid aggregation definition for {agg_name}")
                continue
                
            # Validation récursive de la définition
            validated_def = self._validate_aggregation_definition(agg_def, allowed_fields)
            if validated_def:
                validated[agg_name] = validated_def
        
        logger.debug(f"Validated aggregations: {list(validated.keys())}")
        return validated

    def _validate_aggregation_definition(self, agg_def: Dict[str, Any], allowed_fields: set) -> Optional[Dict[str, Any]]:
        """Valide récursivement une définition d'agrégation."""
        validated = {}
        
        for key, value in agg_def.items():
            # Clé 'field' - valider que le champ existe
            if key == "field" and isinstance(value, str):
                # Retirer .keyword pour la validation
                field_name = value.replace(".keyword", "")
                if field_name in allowed_fields:
                    validated[key] = value
                else:
                    logger.warning(f"Field not allowed in aggregation: {value}")
                    return None  # Rejeter toute l'agrégation
            
            # Clé 'aggs' - validation récursive
            elif key == "aggs" and isinstance(value, dict):
                sub_aggs = {}
                for sub_name, sub_def in value.items():
                    validated_sub = self._validate_aggregation_definition(sub_def, allowed_fields)
                    if validated_sub:
                        sub_aggs[sub_name] = validated_sub
                if sub_aggs:
                    validated[key] = sub_aggs
            
            # Autres clés - passage direct avec validation basique
            else:
                # Limiter les valeurs numériques
                if isinstance(value, (int, float)):
                    if -1000000 <= value <= 1000000:  # Limites raisonnables
                        validated[key] = value
                    else:
                        logger.warning(f"Numeric value out of range: {value}")
                        return None
                
                # Strings courtes seulement
                elif isinstance(value, str) and len(value) <= 100:
                    validated[key] = value
                
                # Dictionnaires et listes - passage direct
                elif isinstance(value, (dict, list)):
                    validated[key] = value
                
                else:
                    logger.warning(f"Invalid value type for key {key}: {type(value)}")
                    return None
        
        return validated if validated else None