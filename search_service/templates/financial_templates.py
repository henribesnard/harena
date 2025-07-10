"""
Templates Financiers - Search Service

Templates spécialisés pour les requêtes financières avec optimisations
pour les transactions, marchands, catégories et analyses de dépenses.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging

from .config import FIELD_GROUPS, FUZZINESS_CONFIG
from .exceptions import InvalidParametersError, TemplateValidationError

logger = logging.getLogger(__name__)


class FinancialQueryTemplates:
    """
    Templates spécialisés pour les requêtes financières avec optimisations
    pour les transactions, marchands, catégories et analyses de dépenses.
    """
    
    @staticmethod
    def merchant_exact_search(
        merchant_name: str,
        user_id: int,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche exacte par nom de marchand (terme exact).
        """
        if not merchant_name.strip():
            raise InvalidParametersError(missing_params=["merchant_name"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        return {
            "bool": {
                "must": [
                    {
                        "term": {
                            "merchant_name.keyword": {
                                "value": merchant_name,
                                "boost": boost
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
    
    @staticmethod
    def merchant_fuzzy_search(
        merchant_name: str,
        user_id: int,
        fuzziness: str = "AUTO:3,6",
        minimum_should_match: str = "75%",
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche floue par nom de marchand avec tolérance aux erreurs.
        """
        if not merchant_name.strip():
            raise InvalidParametersError(missing_params=["merchant_name"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        return {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": merchant_name,
                            "fields": FIELD_GROUPS["merchant_fields"],
                            "type": "best_fields",
                            "fuzziness": fuzziness,
                            "minimum_should_match": minimum_should_match,
                            "boost": boost,
                            "tie_breaker": 0.3
                        }
                    }
                ],
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
    
    @staticmethod
    def category_search_by_id(
        category_id: int,
        user_id: int,
        include_subcategories: bool = False,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche par ID de catégorie avec option sous-catégories.
        """
        if not category_id:
            raise InvalidParametersError(missing_params=["category_id"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        query = {
            "bool": {
                "must": [
                    {
                        "term": {
                            "category_id": {
                                "value": category_id,
                                "boost": boost
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
        
        if include_subcategories:
            query["bool"]["should"] = [
                {"term": {"parent_category_id": category_id}}
            ]
            query["bool"]["minimum_should_match"] = 0
        
        return query
    
    @staticmethod
    def category_search_by_name(
        category_name: str,
        user_id: int,
        include_subcategories: bool = True,
        exact_match: bool = False,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche par nom de catégorie avec options de précision.
        """
        if not category_name.strip():
            raise InvalidParametersError(missing_params=["category_name"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        if exact_match:
            must_clause = {
                "term": {
                    "category_name.keyword": {
                        "value": category_name,
                        "boost": boost
                    }
                }
            }
        else:
            fields = FIELD_GROUPS["category_fields"]
            if not include_subcategories:
                fields = [f for f in fields if "subcategory" not in f]
            
            must_clause = {
                "multi_match": {
                    "query": category_name,
                    "fields": fields,
                    "type": "best_fields",
                    "fuzziness": FUZZINESS_CONFIG["category_fields"],
                    "boost": boost,
                    "tie_breaker": 0.2
                }
            }
        
        return {
            "bool": {
                "must": [must_clause],
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
    
    @staticmethod
    def amount_range_search(
        user_id: int,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        absolute_value: bool = True,
        include_zero: bool = False,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche par plage de montants avec options flexibles.
        """
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        field = "amount_abs" if absolute_value else "amount"
        range_query = {}
        
        if min_amount is not None:
            range_query["gte"] = min_amount
        if max_amount is not None:
            range_query["lte"] = max_amount
        
        must_clauses = []
        if range_query:
            must_clauses.append({
                "range": {
                    field: {
                        **range_query,
                        "boost": boost
                    }
                }
            })
        
        filter_clauses = [{"term": {"user_id": user_id}}]
        
        # Exclure les montants zéro si demandé
        if not include_zero:
            filter_clauses.append({
                "range": {
                    field: {"gt": 0}
                }
            })
        
        return {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    
    @staticmethod
    def date_range_search(
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        date_field: str = "transaction_date",
        timezone: str = "Europe/Paris",
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche par plage de dates avec support timezone.
        """
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        range_query = {}
        
        if start_date:
            range_query["gte"] = start_date.isoformat()
        if end_date:
            range_query["lte"] = end_date.isoformat()
        
        if timezone != "UTC":
            range_query["time_zone"] = timezone
        
        query = {
            "bool": {
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
        
        if range_query:
            query["bool"]["must"] = [
                {
                    "range": {
                        date_field: {
                            **range_query,
                            "boost": boost
                        }
                    }
                }
            ]
        
        return query
    
    @staticmethod
    def spending_analysis_template(
        user_id: int,
        period_start: datetime,
        period_end: datetime,
        categories: Optional[List[int]] = None,
        merchants: Optional[List[str]] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        exclude_categories: Optional[List[int]] = None,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template complexe pour analyse des dépenses avec filtres multiples.
        """
        if not all([user_id, period_start, period_end]):
            raise InvalidParametersError(missing_params=["user_id", "period_start", "period_end"])
        
        must_clauses = [
            {
                "range": {
                    "transaction_date": {
                        "gte": period_start.isoformat(),
                        "lte": period_end.isoformat(),
                        "boost": boost
                    }
                }
            }
        ]
        
        filter_clauses = [{"term": {"user_id": user_id}}]
        must_not_clauses = []
        
        # Filtres de montant
        if min_amount is not None or max_amount is not None:
            amount_range = {}
            if min_amount is not None:
                amount_range["gte"] = min_amount
            if max_amount is not None:
                amount_range["lte"] = max_amount
            
            if amount_range:
                must_clauses.append({"range": {"amount_abs": amount_range}})
        
        # Filtres de catégories
        if categories:
            filter_clauses.append({"terms": {"category_id": categories}})
        
        if exclude_categories:
            must_not_clauses.append({"terms": {"category_id": exclude_categories}})
        
        # Filtres de marchands
        if merchants:
            filter_clauses.append({"terms": {"merchant_name.keyword": merchants}})
        
        query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
        
        if must_not_clauses:
            query["bool"]["must_not"] = must_not_clauses
        
        return query
    
    @staticmethod
    def transaction_search_by_id(
        transaction_id: str,
        user_id: int,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche d'une transaction spécifique par ID.
        """
        if not transaction_id.strip():
            raise InvalidParametersError(missing_params=["transaction_id"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        return {
            "bool": {
                "must": [
                    {
                        "term": {
                            "transaction_id": {
                                "value": transaction_id,
                                "boost": boost
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
    
    @staticmethod
    def recent_transactions_template(
        user_id: int,
        days_back: int = 30,
        limit_amount: Optional[float] = None,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template pour récupérer les transactions récentes.
        """
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        start_date = datetime.now() - timedelta(days=days_back)
        
        must_clauses = [
            {
                "range": {
                    "transaction_date": {
                        "gte": start_date.isoformat(),
                        "boost": boost
                    }
                }
            }
        ]
        
        filter_clauses = [{"term": {"user_id": user_id}}]
        
        if limit_amount:
            must_clauses.append({
                "range": {
                    "amount_abs": {"gte": limit_amount}
                }
            })
        
        return {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    
    @staticmethod
    def multi_merchant_search(
        merchant_names: List[str],
        user_id: int,
        exact_match: bool = False,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Recherche sur plusieurs marchands.
        """
        if not merchant_names:
            raise InvalidParametersError(missing_params=["merchant_names"])
        if not user_id:
            raise InvalidParametersError(missing_params=["user_id"])
        
        if exact_match:
            should_clauses = [
                {"term": {"merchant_name.keyword": merchant}}
                for merchant in merchant_names
            ]
        else:
            should_clauses = [
                {
                    "multi_match": {
                        "query": merchant,
                        "fields": FIELD_GROUPS["merchant_fields"],
                        "fuzziness": FUZZINESS_CONFIG["merchant_fields"]
                    }
                }
                for merchant in merchant_names
            ]
        
        return {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "boost": boost,
                "filter": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
    
    @staticmethod
    def expense_threshold_alert(
        user_id: int,
        threshold_amount: float,
        period_days: int = 30,
        category_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Template pour alertes de dépassement de seuil.
        """
        if not all([user_id, threshold_amount]):
            raise InvalidParametersError(missing_params=["user_id", "threshold_amount"])
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        must_clauses = [
            {
                "range": {
                    "transaction_date": {
                        "gte": start_date.isoformat()
                    }
                }
            },
            {
                "range": {
                    "amount_abs": {
                        "gte": threshold_amount
                    }
                }
            }
        ]
        
        filter_clauses = [{"term": {"user_id": user_id}}]
        
        if category_ids:
            filter_clauses.append({"terms": {"category_id": category_ids}})
        
        return {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }


# ==================== FONCTIONS UTILITAIRES ====================

def validate_financial_params(
    user_id: int,
    amounts: Optional[Dict[str, float]] = None,
    dates: Optional[Dict[str, datetime]] = None,
    **kwargs
) -> bool:
    """
    Valide les paramètres des requêtes financières.
    """
    if not user_id or user_id <= 0:
        raise InvalidParametersError(invalid_params=["user_id"])
    
    if amounts:
        for amount_type, amount_value in amounts.items():
            if amount_value is not None and amount_value < 0:
                raise InvalidParametersError(invalid_params=[f"{amount_type}_amount"])
    
    if dates:
        start_date = dates.get("start")
        end_date = dates.get("end")
        if start_date and end_date and start_date > end_date:
            raise InvalidParametersError(invalid_params=["date_range"])
    
    return True


def create_financial_filter_combination(
    user_id: int,
    filters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combine plusieurs filtres financiers en une requête bool.
    """
    if not user_id:
        raise InvalidParametersError(missing_params=["user_id"])
    
    must_clauses = []
    filter_clauses = [{"term": {"user_id": user_id}}]
    should_clauses = []
    must_not_clauses = []
    
    for filter_config in filters:
        filter_type = filter_config.get("type")
        filter_value = filter_config.get("value")
        filter_operation = filter_config.get("operation", "must")
        
        if filter_type == "amount_range":
            clause = {
                "range": {
                    "amount_abs": {
                        "gte": filter_value.get("min", 0),
                        "lte": filter_value.get("max", float('inf'))
                    }
                }
            }
        elif filter_type == "categories":
            clause = {"terms": {"category_id": filter_value}}
        elif filter_type == "merchants":
            clause = {"terms": {"merchant_name.keyword": filter_value}}
        elif filter_type == "date_range":
            clause = {
                "range": {
                    "transaction_date": {
                        "gte": filter_value.get("start"),
                        "lte": filter_value.get("end")
                    }
                }
            }
        else:
            continue
        
        # Ajouter à la clause appropriée
        if filter_operation == "must":
            must_clauses.append(clause)
        elif filter_operation == "should":
            should_clauses.append(clause)
        elif filter_operation == "must_not":
            must_not_clauses.append(clause)
        elif filter_operation == "filter":
            filter_clauses.append(clause)
    
    query = {"bool": {"filter": filter_clauses}}
    
    if must_clauses:
        query["bool"]["must"] = must_clauses
    if should_clauses:
        query["bool"]["should"] = should_clauses
        query["bool"]["minimum_should_match"] = 1
    if must_not_clauses:
        query["bool"]["must_not"] = must_not_clauses
    
    return query


def optimize_financial_query_performance(
    query: Dict[str, Any],
    user_id: int
) -> Dict[str, Any]:
    """
    Optimise une requête financière pour les performances.
    """
    optimized = query.copy()
    
    # S'assurer que le filtre utilisateur est en mode filter (pas must)
    if "bool" in optimized:
        bool_query = optimized["bool"]
        user_filter = {"term": {"user_id": user_id}}
        
        # Retirer de must si présent
        if "must" in bool_query:
            bool_query["must"] = [
                clause for clause in bool_query["must"] 
                if not (isinstance(clause, dict) and 
                       clause.get("term", {}).get("user_id") == user_id)
            ]
        
        # Ajouter à filter
        if "filter" not in bool_query:
            bool_query["filter"] = []
        
        if user_filter not in bool_query["filter"]:
            bool_query["filter"].append(user_filter)
    
    return optimized


# ==================== EXPORTS ====================

__all__ = [
    "FinancialQueryTemplates",
    "validate_financial_params",
    "create_financial_filter_combination",
    "optimize_financial_query_performance"
]