"""
Query Builder - Search Service

Builder pattern pour construction dynamique de requêtes Elasticsearch complexes.
Permet la composition de requêtes avec validation et optimisation automatique.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from .config import TEMPLATE_CONFIG, FIELD_GROUPS, HIGHLIGHT_CONFIG
from .exceptions import QueryBuilderError, InvalidParametersError
from .text_search import TextSearchTemplates
from ..models.service_contracts import IntentType

logger = logging.getLogger(__name__)


class QueryTemplateBuilder:
    """
    Builder pattern pour construction de requêtes complexes avec validation.
    Permet la composition dynamique de requêtes selon les intentions détectées.
    """
    
    def __init__(self, user_id: Optional[int] = None):
        """Initialise le builder avec un utilisateur optionnel."""
        self._query = {
            "bool": {
                "must": [],
                "should": [],
                "filter": [],
                "must_not": []
            }
        }
        self._sort = []
        self._highlight = {}
        self._source = None
        self._size = TEMPLATE_CONFIG["default_size"]
        self._from = TEMPLATE_CONFIG["default_from"]
        self._boost_functions = []
        self._min_score = None
        self._track_total_hits = TEMPLATE_CONFIG["track_total_hits"]
        self._timeout = TEMPLATE_CONFIG["timeout"]
        self._explain = False
        
        # Ajouter le filtre utilisateur si fourni
        if user_id:
            self.add_user_filter(user_id)
    
    # ==================== CLAUSES BOOL ====================
    
    def add_must_clause(self, clause: Dict[str, Any]) -> 'QueryTemplateBuilder':
        """Ajoute une clause MUST (ET logique)."""
        if not isinstance(clause, dict):
            raise QueryBuilderError("La clause MUST doit être un dictionnaire")
        self._query["bool"]["must"].append(clause)
        return self
    
    def add_should_clause(
        self, 
        clause: Dict[str, Any], 
        minimum_should_match: Optional[Union[int, str]] = None
    ) -> 'QueryTemplateBuilder':
        """Ajoute une clause SHOULD (OU logique)."""
        if not isinstance(clause, dict):
            raise QueryBuilderError("La clause SHOULD doit être un dictionnaire")
        self._query["bool"]["should"].append(clause)
        if minimum_should_match is not None:
            self._query["bool"]["minimum_should_match"] = minimum_should_match
        return self
    
    def add_filter_clause(self, clause: Dict[str, Any]) -> 'QueryTemplateBuilder':
        """Ajoute une clause FILTER (n'affecte pas le score)."""
        if not isinstance(clause, dict):
            raise QueryBuilderError("La clause FILTER doit être un dictionnaire")
        self._query["bool"]["filter"].append(clause)
        return self
    
    def add_must_not_clause(self, clause: Dict[str, Any]) -> 'QueryTemplateBuilder':
        """Ajoute une clause MUST_NOT (exclusion)."""
        if not isinstance(clause, dict):
            raise QueryBuilderError("La clause MUST_NOT doit être un dictionnaire")
        self._query["bool"]["must_not"].append(clause)
        return self
    
    # ==================== RECHERCHES SPÉCIALISÉES ====================
    
    def add_text_search(
        self,
        query_text: str,
        fields: Optional[List[str]] = None,
        search_type: str = "best_fields",
        fuzziness: str = "AUTO",
        boost: float = 1.0
    ) -> 'QueryTemplateBuilder':
        """Ajoute une recherche textuelle multi-champs."""
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not fields:
            fields = FIELD_GROUPS["all_text"]
        
        text_query = TextSearchTemplates.multi_match_best_fields(
            query_text=query_text,
            fields=fields,
            fuzziness=fuzziness,
            boost=boost
        )
        
        self.add_must_clause(text_query)
        return self
    
    def add_merchant_search(
        self,
        merchant_name: str,
        exact_match: bool = False,
        boost: float = 1.0
    ) -> 'QueryTemplateBuilder':
        """Ajoute une recherche par marchand."""
        if not merchant_name.strip():
            raise InvalidParametersError(missing_params=["merchant_name"])
        
        if exact_match:
            merchant_query = {
                "term": {
                    "merchant_name.keyword": {
                        "value": merchant_name,
                        "boost": boost
                    }
                }
            }
        else:
            merchant_query = {
                "multi_match": {
                    "query": merchant_name,
                    "fields": FIELD_GROUPS["merchant_fields"],
                    "type": "best_fields",
                    "fuzziness": "AUTO:3,6",
                    "boost": boost
                }
            }
        
        self.add_must_clause(merchant_query)
        return self
    
    def add_category_filter(
        self,
        category_ids: Optional[List[int]] = None,
        category_names: Optional[List[str]] = None,
        exact_match: bool = True
    ) -> 'QueryTemplateBuilder':
        """Ajoute un filtre par catégorie."""
        if not category_ids and not category_names:
            raise InvalidParametersError(missing_params=["category_ids ou category_names"])
        
        if category_ids:
            self.add_filter_clause({"terms": {"category_id": category_ids}})
        
        if category_names:
            if exact_match:
                self.add_filter_clause({"terms": {"category_name.keyword": category_names}})
            else:
                for category_name in category_names:
                    category_query = {
                        "multi_match": {
                            "query": category_name,
                            "fields": FIELD_GROUPS["category_fields"],
                            "fuzziness": "1"
                        }
                    }
                    self.add_should_clause(category_query)
        
        return self
    
    def add_amount_range(
        self,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        absolute_value: bool = True
    ) -> 'QueryTemplateBuilder':
        """Ajoute un filtre de montant."""
        if min_amount is None and max_amount is None:
            raise InvalidParametersError(missing_params=["min_amount ou max_amount"])
        
        field = "amount_abs" if absolute_value else "amount"
        range_query = {}
        
        if min_amount is not None:
            if min_amount < 0:
                raise InvalidParametersError(invalid_params=["min_amount"])
            range_query["gte"] = min_amount
        
        if max_amount is not None:
            if max_amount < 0:
                raise InvalidParametersError(invalid_params=["max_amount"])
            range_query["lte"] = max_amount
        
        if range_query:
            self.add_filter_clause({"range": {field: range_query}})
        
        return self
    
    def add_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        field: str = "transaction_date",
        timezone: str = "Europe/Paris"
    ) -> 'QueryTemplateBuilder':
        """Ajoute un filtre de date."""
        if start_date is None and end_date is None:
            raise InvalidParametersError(missing_params=["start_date ou end_date"])
        
        if start_date and end_date and start_date > end_date:
            raise InvalidParametersError(invalid_params=["date_range"])
        
        range_query = {}
        
        if start_date:
            range_query["gte"] = start_date.isoformat()
        if end_date:
            range_query["lte"] = end_date.isoformat()
        
        if timezone != "UTC":
            range_query["time_zone"] = timezone
        
        if range_query:
            self.add_filter_clause({"range": {field: range_query}})
        
        return self
    
    def add_user_filter(self, user_id: int) -> 'QueryTemplateBuilder':
        """Ajoute le filtre utilisateur obligatoire."""
        if not user_id or user_id <= 0:
            raise InvalidParametersError(invalid_params=["user_id"])
        self.add_filter_clause({"term": {"user_id": user_id}})
        return self
    
    def add_exists_filter(self, field: str) -> 'QueryTemplateBuilder':
        """Ajoute un filtre d'existence de champ."""
        if not field.strip():
            raise InvalidParametersError(missing_params=["field"])
        self.add_filter_clause({"exists": {"field": field}})
        return self
    
    # ==================== EXCLUSIONS ====================
    
    def exclude_categories(self, category_ids: List[int]) -> 'QueryTemplateBuilder':
        """Exclut certaines catégories."""
        if not category_ids:
            raise InvalidParametersError(missing_params=["category_ids"])
        self.add_must_not_clause({"terms": {"category_id": category_ids}})
        return self
    
    def exclude_merchants(self, merchant_names: List[str]) -> 'QueryTemplateBuilder':
        """Exclut certains marchands."""
        if not merchant_names:
            raise InvalidParametersError(missing_params=["merchant_names"])
        self.add_must_not_clause({"terms": {"merchant_name.keyword": merchant_names}})
        return self
    
    def exclude_amount_range(
        self,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        absolute_value: bool = True
    ) -> 'QueryTemplateBuilder':
        """Exclut une plage de montants."""
        if min_amount is None and max_amount is None:
            raise InvalidParametersError(missing_params=["min_amount ou max_amount"])
        
        field = "amount_abs" if absolute_value else "amount"
        range_query = {}
        
        if min_amount is not None:
            range_query["gte"] = min_amount
        if max_amount is not None:
            range_query["lte"] = max_amount
        
        if range_query:
            self.add_must_not_clause({"range": {field: range_query}})
        
        return self
    
    def exclude_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        field: str = "transaction_date"
    ) -> 'QueryTemplateBuilder':
        """Exclut une plage de dates."""
        if start_date is None and end_date is None:
            raise InvalidParametersError(missing_params=["start_date ou end_date"])
        
        range_query = {}
        
        if start_date:
            range_query["gte"] = start_date.isoformat()
        if end_date:
            range_query["lte"] = end_date.isoformat()
        
        if range_query:
            self.add_must_not_clause({"range": {field: range_query}})
        
        return self
    
    # ==================== TRI ET PAGINATION ====================
    
    def add_sort(
        self,
        field: str,
        order: str = "desc",
        missing: str = "_last",
        unmapped_type: Optional[str] = None
    ) -> 'QueryTemplateBuilder':
        """Ajoute un critère de tri."""
        if not field.strip():
            raise InvalidParametersError(missing_params=["field"])
        
        if order not in ["asc", "desc"]:
            raise InvalidParametersError(invalid_params=["order"])
        
        sort_clause = {field: {"order": order, "missing": missing}}
        if unmapped_type:
            sort_clause[field]["unmapped_type"] = unmapped_type
        
        self._sort.append(sort_clause)
        return self
    
    def add_score_sort(self, order: str = "desc") -> 'QueryTemplateBuilder':
        """Ajoute un tri par score."""
        if order not in ["asc", "desc"]:
            raise InvalidParametersError(invalid_params=["order"])
        self._sort.append({"_score": {"order": order}})
        return self
    
    def set_pagination(self, size: int = 20, from_: int = 0) -> 'QueryTemplateBuilder':
        """Configure la pagination."""
        if size <= 0 or size > TEMPLATE_CONFIG["max_size"]:
            raise InvalidParametersError(invalid_params=["size"])
        if from_ < 0:
            raise InvalidParametersError(invalid_params=["from"])
        
        self._size = size
        self._from = from_
        return self
    
    # ==================== CONFIGURATION RÉSULTATS ====================
    
    def set_source_fields(
        self, 
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ) -> 'QueryTemplateBuilder':
        """Configure les champs à retourner."""
        if include or exclude:
            self._source = {}
            if include:
                self._source["includes"] = include
            if exclude:
                self._source["excludes"] = exclude
        return self
    
    def enable_highlighting(
        self,
        fields: Optional[List[str]] = None,
        pre_tags: Optional[List[str]] = None,
        post_tags: Optional[List[str]] = None,
        fragment_size: int = 150,
        number_of_fragments: int = 3
    ) -> 'QueryTemplateBuilder':
        """Active le highlighting des termes recherchés."""
        if not fields:
            fields = HIGHLIGHT_CONFIG["default_fields"]
        
        if not pre_tags:
            pre_tags = HIGHLIGHT_CONFIG["pre_tags"]
        
        if not post_tags:
            post_tags = HIGHLIGHT_CONFIG["post_tags"]
        
        highlight_fields = {}
        for field in fields:
            highlight_fields[field] = {
                "fragment_size": fragment_size,
                "number_of_fragments": number_of_fragments
            }
        
        self._highlight = {
            "pre_tags": pre_tags,
            "post_tags": post_tags,
            "fields": highlight_fields
        }
        return self
    
    def disable_highlighting(self) -> 'QueryTemplateBuilder':
        """Désactive le highlighting."""
        self._highlight = {}
        return self
    
    def set_timeout(self, timeout: str) -> 'QueryTemplateBuilder':
        """Définit le timeout de la requête."""
        if not timeout:
            raise InvalidParametersError(missing_params=["timeout"])
        self._timeout = timeout
        return self
    
    def set_min_score(self, min_score: float) -> 'QueryTemplateBuilder':
        """Définit le score minimum requis."""
        if min_score < 0:
            raise InvalidParametersError(invalid_params=["min_score"])
        self._min_score = min_score
        return self
    
    def enable_explain(self) -> 'QueryTemplateBuilder':
        """Active l'explication du scoring."""
        self._explain = True
        return self
    
    def disable_explain(self) -> 'QueryTemplateBuilder':
        """Désactive l'explication du scoring."""
        self._explain = False
        return self
    
    # ==================== FONCTIONS DE BOOST ====================
    
    def add_boost_function(
        self,
        function_type: str,
        field: Optional[str] = None,
        weight: float = 1.0,
        **params
    ) -> 'QueryTemplateBuilder':
        """Ajoute une fonction de boost."""
        if weight <= 0:
            raise InvalidParametersError(invalid_params=["weight"])
        
        if function_type == "field_value_factor":
            if not field:
                raise InvalidParametersError(missing_params=["field"])
            
            boost_func = {
                "field_value_factor": {
                    "field": field,
                    "factor": params.get("factor", 1.0),
                    "modifier": params.get("modifier", "none"),
                    "missing": params.get("missing", 1.0)
                },
                "weight": weight
            }
        
        elif function_type in ["gauss", "linear", "exp"]:
            if not field:
                raise InvalidParametersError(missing_params=["field"])
            
            boost_func = {
                function_type: {
                    field: {
                        "origin": params.get("origin"),
                        "scale": params.get("scale"),
                        "offset": params.get("offset", 0),
                        "decay": params.get("decay", 0.5)
                    }
                },
                "weight": weight
            }
        
        elif function_type == "script_score":
            script = params.get("script")
            if not script:
                raise InvalidParametersError(missing_params=["script"])
            
            boost_func = {
                "script_score": {
                    "script": script
                },
                "weight": weight
            }
        
        elif function_type == "filter":
            filter_clause = params.get("filter")
            if not filter_clause:
                raise InvalidParametersError(missing_params=["filter"])
            
            boost_func = {
                "filter": filter_clause,
                "weight": weight
            }
        
        else:
            raise InvalidParametersError(invalid_params=["function_type"])
        
        self._boost_functions.append(boost_func)
        return self
    
    def add_recency_boost(
        self,
        date_field: str = "transaction_date",
        origin: str = "now",
        scale: str = "30d",
        decay: float = 0.5,
        weight: float = 2.0
    ) -> 'QueryTemplateBuilder':
        """Ajoute un boost de récence."""
        return self.add_boost_function(
            function_type="gauss",
            field=date_field,
            weight=weight,
            origin=origin,
            scale=scale,
            decay=decay
        )
    
    def add_amount_boost(
        self,
        amount_field: str = "amount_abs",
        factor: float = 0.1,
        modifier: str = "log1p",
        weight: float = 1.5
    ) -> 'QueryTemplateBuilder':
        """Ajoute un boost basé sur le montant."""
        return self.add_boost_function(
            function_type="field_value_factor",
            field=amount_field,
            weight=weight,
            factor=factor,
            modifier=modifier,
            missing=1.0
        )
    
    def clear_boost_functions(self) -> 'QueryTemplateBuilder':
        """Supprime toutes les fonctions de boost."""
        self._boost_functions = []
        return self
    
    # ==================== OPTIMISATIONS PAR INTENTION ====================
    
    def optimize_for_intent(self, intent_type: IntentType) -> 'QueryTemplateBuilder':
        """Optimise la requête selon le type d'intention."""
        if intent_type == IntentType.SPENDING_ANALYSIS:
            self.add_sort("amount_abs", "desc")
            self.add_sort("transaction_date", "desc")
            self.add_amount_boost()
            self.set_pagination(size=50)
            
        elif intent_type == IntentType.MERCHANT_SEARCH:
            self.add_score_sort("desc")
            self.add_sort("transaction_date", "desc")
            self.enable_highlighting(fields=["merchant_name", "merchant_alias"])
            
        elif intent_type == IntentType.CATEGORY_SEARCH:
            self.add_score_sort("desc")
            self.add_sort("amount_abs", "desc")
            self.enable_highlighting(fields=["category_name", "subcategory_name"])
            
        elif intent_type == IntentType.RECENT_TRANSACTIONS:
            self.add_sort("transaction_date", "desc")
            self.add_score_sort("desc")
            self.add_recency_boost()
            self.set_pagination(size=30)
            
        elif intent_type == IntentType.TEXT_SEARCH:
            self.add_score_sort("desc")
            self.enable_highlighting()
            
        elif intent_type == IntentType.SPENDING_EVOLUTION:
            self.add_sort("transaction_date", "asc")
            self.add_sort("amount_abs", "desc")
            self.set_pagination(size=100)
        
        return self
    
    # ==================== CONSTRUCTION ET VALIDATION ====================
    
    def validate(self) -> bool:
        """Valide la requête construite."""
        bool_query = self._query["bool"]
        
        # Vérifier qu'il y a au moins une clause de recherche
        has_clauses = any(bool_query[clause_type] for clause_type in ["must", "should", "filter"])
        if not has_clauses:
            raise QueryBuilderError("La requête doit contenir au moins une clause de recherche")
        
        # Vérifier la présence du filtre utilisateur
        user_filter_present = any(
            isinstance(clause, dict) and 
            clause.get("term", {}).get("user_id") is not None
            for clause in bool_query.get("filter", [])
        )
        
        if not user_filter_present:
            logger.warning("Aucun filtre utilisateur détecté - potentiel problème de sécurité")
        
        # Valider les paramètres de pagination
        if self._from + self._size > 10000:
            logger.warning(f"Pagination profonde détectée (from: {self._from}, size: {self._size})")
        
        return True
    
    def build(self) -> Dict[str, Any]:
        """Construit la requête finale Elasticsearch."""
        # Valider avant construction
        self.validate()
        
        # Nettoyer les clauses vides
        cleaned_bool = {}
        for clause_type, clauses in self._query["bool"].items():
            if clauses:
                cleaned_bool[clause_type] = clauses
        
        # Construire la requête de base
        if not cleaned_bool:
            final_query = {"match_all": {}}
        else:
            final_query = {"bool": cleaned_bool}
        
        # Appliquer les fonctions de boost si nécessaire
        if self._boost_functions:
            final_query = {
                "function_score": {
                    "query": final_query,
                    "functions": self._boost_functions,
                    "score_mode": "multiply",
                    "boost_mode": "multiply"
                }
            }
        
        # Construire la requête Elasticsearch complète
        elasticsearch_query = {
            "query": final_query,
            "size": self._size,
            "from": self._from,
            "track_total_hits": self._track_total_hits,
            "timeout": self._timeout
        }
        
        # Ajouter le tri si spécifié
        if self._sort:
            elasticsearch_query["sort"] = self._sort
        
        # Ajouter les champs source si spécifiés
        if self._source is not None:
            elasticsearch_query["_source"] = self._source
        
        # Ajouter le highlighting si activé
        if self._highlight:
            elasticsearch_query["highlight"] = self._highlight
        
        # Ajouter le score minimum si spécifié
        if self._min_score is not None:
            elasticsearch_query["min_score"] = self._min_score
        
        # Ajouter l'explication si activée
        if self._explain:
            elasticsearch_query["explain"] = True
        
        return elasticsearch_query
    
    def reset(self) -> 'QueryTemplateBuilder':
        """Remet à zéro le builder."""
        self.__init__()
        return self
    
    def clone(self) -> 'QueryTemplateBuilder':
        """Crée une copie du builder."""
        cloned = QueryTemplateBuilder()
        cloned._query = {
            "bool": {
                "must": self._query["bool"]["must"].copy(),
                "should": self._query["bool"]["should"].copy(),
                "filter": self._query["bool"]["filter"].copy(),
                "must_not": self._query["bool"]["must_not"].copy()
            }
        }
        cloned._sort = self._sort.copy()
        cloned._highlight = self._highlight.copy()
        cloned._source = self._source
        cloned._size = self._size
        cloned._from = self._from
        cloned._boost_functions = self._boost_functions.copy()
        cloned._min_score = self._min_score
        cloned._track_total_hits = self._track_total_hits
        cloned._timeout = self._timeout
        cloned._explain = self._explain
        return cloned
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur la requête construite."""
        bool_query = self._query["bool"]
        return {
            "must_clauses": len(bool_query.get("must", [])),
            "should_clauses": len(bool_query.get("should", [])),
            "filter_clauses": len(bool_query.get("filter", [])),
            "must_not_clauses": len(bool_query.get("must_not", [])),
            "boost_functions": len(self._boost_functions),
            "sort_criteria": len(self._sort),
            "has_highlighting": bool(self._highlight),
            "has_source_filtering": self._source is not None,
            "pagination": {
                "size": self._size,
                "from": self._from
            }
        }


# ==================== FONCTIONS UTILITAIRES ====================

def create_quick_query(
    user_id: int,
    query_text: Optional[str] = None,
    intent_type: Optional[IntentType] = None,
    **filters
) -> Dict[str, Any]:
    """Crée rapidement une requête avec les paramètres de base."""
    builder = QueryTemplateBuilder(user_id=user_id)
    
    # Ajouter recherche textuelle si fournie
    if query_text:
        builder.add_text_search(query_text)
    
    # Ajouter filtres
    if "merchant_name" in filters:
        builder.add_merchant_search(filters["merchant_name"])
    
    if "category_ids" in filters:
        builder.add_category_filter(category_ids=filters["category_ids"])
    
    if "min_amount" in filters or "max_amount" in filters:
        builder.add_amount_range(
            min_amount=filters.get("min_amount"),
            max_amount=filters.get("max_amount")
        )
    
    if "start_date" in filters or "end_date" in filters:
        builder.add_date_range(
            start_date=filters.get("start_date"),
            end_date=filters.get("end_date")
        )
    
    # Optimiser pour l'intention si fournie
    if intent_type:
        builder.optimize_for_intent(intent_type)
    
    return builder.build()


def validate_builder_query(query: Dict[str, Any]) -> bool:
    """Valide une requête générée par le builder."""
    try:
        # Vérifications de base
        if not isinstance(query, dict):
            return False
        
        required_keys = ["query", "size", "from"]
        if not all(key in query for key in required_keys):
            return False
        
        # Vérifier la structure de la requête
        query_part = query["query"]
        if not isinstance(query_part, dict):
            return False
        
        # Vérifier les valeurs de pagination
        if query["size"] <= 0 or query["from"] < 0:
            return False
        
        # Vérifier la structure bool si présente
        if "bool" in query_part:
            bool_query = query_part["bool"]
            if not isinstance(bool_query, dict):
                return False
            
            valid_bool_keys = {"must", "should", "filter", "must_not", "minimum_should_match", "boost"}
            if not set(bool_query.keys()).issubset(valid_bool_keys):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur validation requête builder: {e}")
        return False


def optimize_builder_performance(builder: QueryTemplateBuilder) -> QueryTemplateBuilder:
    """Optimise un builder pour les performances."""
    optimized = builder.clone()
    
    # Limiter le nombre de clauses should
    should_clauses = optimized._query["bool"]["should"]
    if len(should_clauses) > 10:
        logger.warning(f"Nombre élevé de clauses should: {len(should_clauses)}")
        optimized._query["bool"]["should"] = should_clauses[:10]
    
    # Limiter le nombre de fonctions de boost
    if len(optimized._boost_functions) > 5:
        logger.warning(f"Nombre élevé de fonctions boost: {len(optimized._boost_functions)}")
        optimized._boost_functions = optimized._boost_functions[:5]
    
    # Optimiser la pagination
    if optimized._size > 100:
        logger.info(f"Réduction de la taille de pagination de {optimized._size} à 100")
        optimized._size = 100
    
    return optimized


# ==================== EXPORTS ====================

__all__ = [
    "QueryTemplateBuilder",
    "create_quick_query",
    "validate_builder_query",
    "optimize_builder_performance"
]