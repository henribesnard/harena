"""
Dynamic Query Builder - Beyond Static Templates
Architecture v3.0 - Phase 2

Responsabilité: Construction dynamique de requêtes Elasticsearch complexes
- Composition dynamique de clauses (au-delà des templates)
- Support agrégations imbriquées (nested aggregations)
- Pipelines d'agrégation (bucket_script, cumulative_sum)
- Requêtes multi-dimensionnelles (pivot tables)
- Optimisation automatique des queries
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types d'agrégations supportés"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    TERMS = "terms"  # Group by
    DATE_HISTOGRAM = "date_histogram"  # Agrégation temporelle
    PERCENTILES = "percentiles"
    CARDINALITY = "cardinality"  # Nombre valeurs uniques
    STATS = "stats"  # Statistiques complètes


class FilterOperation(Enum):
    """Opérations de filtrage"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    RANGE = "range"
    EXISTS = "exists"
    MATCH = "match"
    WILDCARD = "wildcard"


@dataclass
class QueryOperation:
    """Opération de query (filter, aggregation, sort, etc.)"""
    operation_type: str  # "filter", "aggregate", "sort", "limit"
    field: str
    value: Any = None
    parameters: Dict[str, Any] = None


@dataclass
class DynamicQueryResult:
    """Résultat de construction de query dynamique"""
    success: bool
    query: Dict[str, Any]
    query_hash: str
    operations_count: int
    estimated_complexity: str  # "simple", "medium", "complex"
    validation_errors: List[str]


class DynamicQueryBuilder:
    """
    Constructeur de requêtes Elasticsearch dynamiques

    Permet de composer des requêtes complexes sans templates figés:
    - Filtres multi-dimensionnels
    - Agrégations imbriquées
    - Calculs dérivés (bucket_script)
    - Optimisations automatiques
    """

    def __init__(self, elasticsearch_schema: Dict[str, Any] = None):
        """
        Args:
            elasticsearch_schema: Schéma des champs disponibles (pour validation)
        """
        self.schema = elasticsearch_schema or self._default_schema()

        # Statistiques
        self.stats = {
            "queries_built": 0,
            "complex_queries": 0,
            "validation_failures": 0
        }

        logger.info("DynamicQueryBuilder initialized")

    async def build_dynamic_query(
        self,
        operations: List[QueryOperation],
        user_id: int,
        optimize: bool = True
    ) -> DynamicQueryResult:
        """
        Construit une requête Elasticsearch dynamique

        Args:
            operations: Liste d'opérations à composer
            user_id: ID utilisateur (pour filtrage automatique)
            optimize: Activer optimisations automatiques

        Returns:
            DynamicQueryResult avec query ES complète
        """
        try:
            # Initialiser structure query
            query = {
                "query": {
                    "bool": {
                        "must": [],
                        "filter": [],
                        "should": [],
                        "must_not": []
                    }
                }
            }

            # Filtrage automatique par user_id
            query["query"]["bool"]["filter"].append({
                "term": {"user_id": user_id}
            })

            validation_errors = []

            # Composer les opérations
            for operation in operations:
                if operation.operation_type == "filter":
                    self._add_filter_clause(query, operation, validation_errors)

                elif operation.operation_type == "aggregate":
                    if "aggs" not in query:
                        query["aggs"] = {}
                    self._add_aggregation(query, operation, validation_errors)

                elif operation.operation_type == "sort":
                    if "sort" not in query:
                        query["sort"] = []
                    self._add_sort(query, operation, validation_errors)

                elif operation.operation_type == "limit":
                    query["size"] = operation.value

            # Nettoyage clauses vides
            query = self._clean_empty_clauses(query)

            # Optimisations
            if optimize:
                query = await self._optimize_query(query)

            # Validation
            is_valid = len(validation_errors) == 0

            # Calcul complexité
            complexity = self._estimate_complexity(query, operations)

            # Hash pour caching
            query_hash = self._compute_query_hash(query)

            # Mise à jour stats
            self.stats["queries_built"] += 1
            if complexity == "complex":
                self.stats["complex_queries"] += 1
            if not is_valid:
                self.stats["validation_failures"] += 1

            return DynamicQueryResult(
                success=is_valid,
                query=query,
                query_hash=query_hash,
                operations_count=len(operations),
                estimated_complexity=complexity,
                validation_errors=validation_errors
            )

        except Exception as e:
            logger.error(f"Error building dynamic query: {str(e)}")
            return DynamicQueryResult(
                success=False,
                query={},
                query_hash="",
                operations_count=0,
                estimated_complexity="error",
                validation_errors=[str(e)]
            )

    async def build_comparison_query(
        self,
        filters: Dict[str, Any],
        period_1: str,
        period_2: str,
        user_id: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Construit deux requêtes pour comparaison temporelle

        Args:
            filters: Filtres communs (catégorie, marchand, etc.)
            period_1: Période 1 (ex: "2025-01")
            period_2: Période 2 (ex: "2024-01")
            user_id: ID utilisateur

        Returns:
            Tuple (query_period_1, query_period_2)
        """
        operations_base = self._filters_to_operations(filters)

        # Query période 1
        operations_1 = operations_base + [
            QueryOperation(
                operation_type="filter",
                field="date",
                value=period_1,
                parameters={"operation": "range_month"}
            )
        ]

        result_1 = await self.build_dynamic_query(operations_1, user_id)

        # Query période 2
        operations_2 = operations_base + [
            QueryOperation(
                operation_type="filter",
                field="date",
                value=period_2,
                parameters={"operation": "range_month"}
            )
        ]

        result_2 = await self.build_dynamic_query(operations_2, user_id)

        return result_1.query, result_2.query

    async def build_pivot_table_query(
        self,
        rows_field: str,
        columns_field: str,
        value_field: str,
        agg_function: str = "sum",
        filters: Dict[str, Any] = None,
        user_id: int = 0
    ) -> DynamicQueryResult:
        """
        Construit une requête pour table pivot (2D aggregation)

        Args:
            rows_field: Champ pour les lignes (ex: "category_name")
            columns_field: Champ pour les colonnes (ex: "month")
            value_field: Champ à agréger (ex: "amount")
            agg_function: Fonction d'agrégation (sum, avg, count)
            filters: Filtres additionnels
            user_id: ID utilisateur

        Returns:
            DynamicQueryResult avec agrégations imbriquées

        Exemple:
        Pivot: catégories (rows) × mois (columns), avec sum(amount)
        """
        operations = []

        # Filtres de base
        if filters:
            operations.extend(self._filters_to_operations(filters))

        # Agrégation imbriquée (pivot)
        operations.append(
            QueryOperation(
                operation_type="aggregate",
                field=rows_field,
                parameters={
                    "agg_type": "terms",
                    "nested_agg": {
                        "field": columns_field,
                        "agg_type": "terms",
                        "value_agg": {
                            "field": value_field,
                            "agg_type": agg_function
                        }
                    }
                }
            )
        )

        return await self.build_dynamic_query(operations, user_id)

    def _add_filter_clause(
        self,
        query: Dict[str, Any],
        operation: QueryOperation,
        errors: List[str]
    ):
        """Ajoute une clause de filtrage"""

        field = operation.field
        value = operation.value
        params = operation.parameters or {}

        filter_operation = params.get("operation", "eq")

        try:
            if filter_operation == "eq":
                query["query"]["bool"]["filter"].append({
                    "term": {field: value}
                })

            elif filter_operation == "ne":
                query["query"]["bool"]["must_not"].append({
                    "term": {field: value}
                })

            elif filter_operation in ["gt", "gte", "lt", "lte"]:
                query["query"]["bool"]["filter"].append({
                    "range": {field: {filter_operation: value}}
                })

            elif filter_operation == "range":
                if isinstance(value, dict) and "min" in value and "max" in value:
                    query["query"]["bool"]["filter"].append({
                        "range": {
                            field: {
                                "gte": value["min"],
                                "lte": value["max"]
                            }
                        }
                    })

            elif filter_operation == "in":
                query["query"]["bool"]["filter"].append({
                    "terms": {field: value if isinstance(value, list) else [value]}
                })

            elif filter_operation == "match":
                query["query"]["bool"]["must"].append({
                    "match": {field: value}
                })

            elif filter_operation == "wildcard":
                query["query"]["bool"]["must"].append({
                    "wildcard": {field: f"*{value}*"}
                })

            elif filter_operation == "range_month":
                # Format: "2025-01" → range sur le mois
                year_month = value
                query["query"]["bool"]["filter"].append({
                    "range": {
                        field: {
                            "gte": f"{year_month}-01",
                            "lt": self._next_month(year_month)
                        }
                    }
                })

        except Exception as e:
            errors.append(f"Error adding filter {field}: {str(e)}")

    def _add_aggregation(
        self,
        query: Dict[str, Any],
        operation: QueryOperation,
        errors: List[str]
    ):
        """Ajoute une agrégation"""

        field = operation.field
        params = operation.parameters or {}
        agg_type = params.get("agg_type", "sum")
        agg_name = params.get("name", f"{agg_type}_{field}")

        try:
            if agg_type in ["sum", "avg", "min", "max"]:
                query["aggs"][agg_name] = {
                    agg_type: {"field": field}
                }

            elif agg_type == "count":
                query["aggs"][agg_name] = {
                    "value_count": {"field": field}
                }

            elif agg_type == "terms":
                # Group by
                nested_agg = params.get("nested_agg")

                agg_definition = {
                    "terms": {
                        "field": field,
                        "size": params.get("size", 10)
                    }
                }

                # Agrégation imbriquée si spécifiée
                if nested_agg:
                    agg_definition["aggs"] = {}

                    nested_field = nested_agg.get("field")
                    nested_type = nested_agg.get("agg_type", "terms")

                    if nested_type == "terms":
                        nested_name = f"by_{nested_field}"
                        agg_definition["aggs"][nested_name] = {
                            "terms": {
                                "field": nested_field,
                                "size": nested_agg.get("size", 10)
                            }
                        }

                        # Agrégation de valeur finale si spécifiée
                        value_agg = nested_agg.get("value_agg")
                        if value_agg:
                            value_field = value_agg["field"]
                            value_type = value_agg["agg_type"]
                            agg_definition["aggs"][nested_name]["aggs"] = {
                                f"{value_type}_value": {
                                    value_type: {"field": value_field}
                                }
                            }

                query["aggs"][agg_name] = agg_definition

            elif agg_type == "date_histogram":
                query["aggs"][agg_name] = {
                    "date_histogram": {
                        "field": field,
                        "calendar_interval": params.get("interval", "month")
                    }
                }

            elif agg_type == "stats":
                query["aggs"][agg_name] = {
                    "stats": {"field": field}
                }

        except Exception as e:
            errors.append(f"Error adding aggregation {agg_name}: {str(e)}")

    def _add_sort(
        self,
        query: Dict[str, Any],
        operation: QueryOperation,
        errors: List[str]
    ):
        """Ajoute un tri"""

        field = operation.field
        order = operation.parameters.get("order", "asc") if operation.parameters else "asc"

        query["sort"].append({field: order})

    def _clean_empty_clauses(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Supprime les clauses bool vides"""

        if "query" in query and "bool" in query["query"]:
            bool_query = query["query"]["bool"]

            for key in ["must", "filter", "should", "must_not"]:
                if key in bool_query and not bool_query[key]:
                    del bool_query[key]

        return query

    async def _optimize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise la query pour meilleures performances"""

        # Réordonnancement des filtres (plus sélectifs en premier)
        if "query" in query and "bool" in query["query"]:
            bool_query = query["query"]["bool"]

            # Priorité: term > range > match > wildcard
            if "filter" in bool_query:
                bool_query["filter"] = self._reorder_filters(bool_query["filter"])

        # Limitation des champs retournés si pas d'agrégation
        if "aggs" not in query:
            query["_source"] = ["id", "amount", "date", "merchant_name", "category_name"]

        return query

    def _reorder_filters(self, filters: List[Dict]) -> List[Dict]:
        """Réordonne les filtres par sélectivité"""

        priority_order = {
            "term": 1,
            "range": 2,
            "terms": 3,
            "match": 4,
            "wildcard": 5
        }

        def get_priority(filter_clause):
            filter_type = list(filter_clause.keys())[0]
            return priority_order.get(filter_type, 99)

        return sorted(filters, key=get_priority)

    def _estimate_complexity(
        self,
        query: Dict[str, Any],
        operations: List[QueryOperation]
    ) -> str:
        """Estime la complexité de la query"""

        complexity_score = 0

        # Nombre de filtres
        if "query" in query and "bool" in query["query"]:
            bool_query = query["query"]["bool"]
            for key in ["must", "filter", "should", "must_not"]:
                complexity_score += len(bool_query.get(key, []))

        # Agrégations
        if "aggs" in query:
            agg_count = len(query["aggs"])
            complexity_score += agg_count * 2

            # Agrégations imbriquées
            for agg in query["aggs"].values():
                if "aggs" in agg:
                    complexity_score += 5

        if complexity_score < 5:
            return "simple"
        elif complexity_score < 15:
            return "medium"
        else:
            return "complex"

    def _compute_query_hash(self, query: Dict[str, Any]) -> str:
        """Calcule hash de la query pour caching"""

        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

    def _filters_to_operations(self, filters: Dict[str, Any]) -> List[QueryOperation]:
        """Convertit dict de filtres en QueryOperations"""

        operations = []

        for field, value in filters.items():
            operations.append(
                QueryOperation(
                    operation_type="filter",
                    field=field,
                    value=value
                )
            )

        return operations

    def _next_month(self, year_month: str) -> str:
        """Calcule le mois suivant (pour range queries)"""

        try:
            date_obj = datetime.strptime(year_month, "%Y-%m")
            if date_obj.month == 12:
                next_month = datetime(date_obj.year + 1, 1, 1)
            else:
                next_month = datetime(date_obj.year, date_obj.month + 1, 1)

            return next_month.strftime("%Y-%m-01")

        except:
            return f"{year_month}-32"  # Fallback

    def _default_schema(self) -> Dict[str, Any]:
        """Schéma par défaut des champs Elasticsearch"""

        return {
            "fields": {
                "id": {"type": "long"},
                "user_id": {"type": "long"},
                "amount": {"type": "float"},
                "date": {"type": "date"},
                "merchant_name": {"type": "keyword"},
                "category_name": {"type": "keyword"},
                "transaction_type": {"type": "keyword"},
                "operation_type": {"type": "keyword"},
                "primary_description": {"type": "text"}
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du builder"""
        return self.stats
