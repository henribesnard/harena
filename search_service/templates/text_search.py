"""
Templates de Recherche Textuelle - Search Service

Templates optimisés pour la recherche textuelle multi-champs avec scoring BM25.
Supporte différentes stratégies de matching (best_fields, cross_fields, phrase, etc.).
"""

from typing import Dict, Any, List, Optional, Union
import logging

from .config import FIELD_GROUPS, FUZZINESS_CONFIG, PERFORMANCE_CONFIG
from .exceptions import TemplateValidationError, InvalidParametersError

logger = logging.getLogger(__name__)


class TextSearchTemplates:
    """
    Templates optimisés pour la recherche textuelle multi-champs.
    Implémente différentes stratégies de matching avec scoring BM25 avancé.
    """
    
    @staticmethod
    def multi_match_best_fields(
        query_text: str,
        fields: Optional[List[str]] = None,
        fuzziness: str = "AUTO",
        tie_breaker: float = 0.3,
        operator: str = "or",
        minimum_should_match: str = "75%",
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template multi_match type best_fields pour recherche textuelle optimisée.
        
        Args:
            query_text: Texte à rechercher
            fields: Liste des champs avec boosts
            fuzziness: Niveau de fuzziness (AUTO, 0, 1, 2)
            tie_breaker: Coefficient pour combiner les scores
            operator: Opérateur logique (and/or)
            minimum_should_match: Pourcentage minimum de termes qui doivent matcher
            boost: Boost global de la requête
            
        Returns:
            Template de requête multi_match optimisé
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not fields:
            fields = FIELD_GROUPS["all_text"]
        
        template = {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "best_fields",
                "fuzziness": fuzziness,
                "tie_breaker": tie_breaker,
                "operator": operator,
                "minimum_should_match": minimum_should_match,
                "auto_generate_synonyms_phrase_query": False,
                "fuzzy_transpositions": True,
                "lenient": True
            }
        }
        
        if boost != 1.0:
            template["multi_match"]["boost"] = boost
            
        return template
    
    @staticmethod
    def multi_match_cross_fields(
        query_text: str,
        fields: Optional[List[str]] = None,
        operator: str = "and",
        tie_breaker: float = 0.0,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template multi_match type cross_fields pour recherche entre champs.
        Optimal pour recherches de noms complets ou expressions multi-mots.
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not fields:
            fields = FIELD_GROUPS["primary_text"]
        
        return {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "cross_fields",
                "operator": operator,
                "tie_breaker": tie_breaker,
                "boost": boost,
                "analyzer": "standard"
            }
        }
    
    @staticmethod
    def phrase_search(
        query_text: str,
        fields: Optional[List[str]] = None,
        slop: int = 2,
        boost: float = 1.5
    ) -> Dict[str, Any]:
        """
        Template pour recherche de phrases avec distance flexible.
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not fields:
            fields = FIELD_GROUPS["description_fields"]
        
        return {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "phrase",
                "slop": slop,
                "boost": boost
            }
        }
    
    @staticmethod
    def phrase_prefix_search(
        query_text: str,
        fields: Optional[List[str]] = None,
        slop: int = 2,
        max_expansions: int = 50,
        boost: float = 1.2
    ) -> Dict[str, Any]:
        """
        Template pour recherche de phrase avec préfixe (autocomplétion).
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not fields:
            fields = FIELD_GROUPS["primary_text"]
        
        return {
            "multi_match": {
                "query": query_text,
                "fields": fields,
                "type": "phrase_prefix",
                "slop": slop,
                "max_expansions": max_expansions,
                "boost": boost
            }
        }
    
    @staticmethod
    def fuzzy_search(
        query_text: str,
        field: str = "searchable_text",
        fuzziness: Union[str, int] = "AUTO",
        max_expansions: int = 50,
        prefix_length: int = 1,
        boost: float = 0.8
    ) -> Dict[str, Any]:
        """
        Template pour recherche floue sur un champ spécifique.
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        return {
            "fuzzy": {
                field: {
                    "value": query_text,
                    "fuzziness": fuzziness,
                    "max_expansions": max_expansions,
                    "prefix_length": prefix_length,
                    "transpositions": True,
                    "boost": boost
                }
            }
        }
    
    @staticmethod
    def wildcard_search(
        pattern: str,
        field: str = "searchable_text",
        boost: float = 1.0,
        case_insensitive: bool = True
    ) -> Dict[str, Any]:
        """
        Template pour recherche avec wildcards (* et ?).
        """
        if not pattern.strip():
            raise InvalidParametersError(missing_params=["pattern"])
        
        return {
            "wildcard": {
                field: {
                    "value": pattern,
                    "boost": boost,
                    "case_insensitive": case_insensitive
                }
            }
        }
    
    @staticmethod
    def prefix_search(
        prefix: str,
        field: str = "merchant_name",
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template pour recherche par préfixe.
        """
        if not prefix.strip():
            raise InvalidParametersError(missing_params=["prefix"])
        
        return {
            "prefix": {
                field: {
                    "value": prefix,
                    "boost": boost
                }
            }
        }
    
    @staticmethod
    def regexp_search(
        regex_pattern: str,
        field: str = "searchable_text",
        flags: str = "INTERSECTION|COMPLEMENT|EMPTY",
        max_determinized_states: int = 10000,
        boost: float = 1.0
    ) -> Dict[str, Any]:
        """
        Template pour recherche par expression régulière.
        """
        if not regex_pattern.strip():
            raise InvalidParametersError(missing_params=["regex_pattern"])
        
        return {
            "regexp": {
                field: {
                    "value": regex_pattern,
                    "flags": flags,
                    "max_determinized_states": max_determinized_states,
                    "boost": boost
                }
            }
        }
    
    @staticmethod
    def match_all() -> Dict[str, Any]:
        """Template pour matcher tous les documents."""
        return {"match_all": {}}
    
    @staticmethod
    def exists_search(field: str) -> Dict[str, Any]:
        """Template pour vérifier l'existence d'un champ."""
        return {"exists": {"field": field}}
    
    @staticmethod
    def combined_text_search(
        query_text: str,
        primary_fields: Optional[List[str]] = None,
        secondary_fields: Optional[List[str]] = None,
        primary_boost: float = 2.0,
        secondary_boost: float = 1.0,
        fuzziness: str = "AUTO"
    ) -> Dict[str, Any]:
        """
        Template pour recherche textuelle combinée avec priorités.
        """
        if not query_text.strip():
            raise InvalidParametersError(missing_params=["query_text"])
        
        if not primary_fields:
            primary_fields = FIELD_GROUPS["primary_text"]
        if not secondary_fields:
            secondary_fields = FIELD_GROUPS["description_fields"]
        
        return {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": primary_fields,
                            "type": "best_fields",
                            "fuzziness": fuzziness,
                            "boost": primary_boost
                        }
                    },
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": secondary_fields,
                            "type": "best_fields",
                            "fuzziness": fuzziness,
                            "boost": secondary_boost
                        }
                    }
                ]
            }
        }


# ==================== FONCTIONS UTILITAIRES ====================

def validate_text_query_params(
    query_text: str,
    fields: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """
    Valide les paramètres d'une requête textuelle.
    """
    if not query_text or not query_text.strip():
        raise InvalidParametersError(missing_params=["query_text"])
    
    if fields and not isinstance(fields, list):
        raise InvalidParametersError(invalid_params=["fields"])
    
    # Vérifier les paramètres de performance
    max_expansions = kwargs.get("max_expansions", 50)
    if max_expansions > PERFORMANCE_CONFIG["max_expansions"]:
        logger.warning(f"max_expansions {max_expansions} dépasse la limite recommandée")
    
    return True


def optimize_text_search_for_performance(
    template: Dict[str, Any],
    max_expansions_limit: int = 50
) -> Dict[str, Any]:
    """
    Optimise un template de recherche textuelle pour les performances.
    """
    optimized = template.copy()
    
    # Limiter max_expansions
    def limit_expansions(query_dict):
        if isinstance(query_dict, dict):
            for key, value in query_dict.items():
                if key == "max_expansions" and isinstance(value, int):
                    query_dict[key] = min(value, max_expansions_limit)
                elif isinstance(value, dict):
                    limit_expansions(value)
    
    limit_expansions(optimized)
    return optimized


def create_adaptive_text_search(
    query_text: str,
    query_length: Optional[int] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crée une recherche textuelle adaptative selon la longueur et le contexte.
    """
    if query_length is None:
        query_length = len(query_text.split())
    
    # Stratégie selon la longueur de la requête
    if query_length == 1:
        # Mot unique : utiliser fuzzy + prefix
        return {
            "bool": {
                "should": [
                    TextSearchTemplates.multi_match_best_fields(query_text, boost=2.0),
                    TextSearchTemplates.fuzzy_search(query_text, boost=1.0),
                    TextSearchTemplates.prefix_search(query_text, boost=1.5)
                ],
                "minimum_should_match": 1
            }
        }
    
    elif query_length <= 3:
        # Phrase courte : best_fields + phrase
        return {
            "bool": {
                "should": [
                    TextSearchTemplates.multi_match_best_fields(query_text, boost=2.0),
                    TextSearchTemplates.phrase_search(query_text, boost=1.5)
                ],
                "minimum_should_match": 1
            }
        }
    
    else:
        # Phrase longue : cross_fields + phrase avec slop
        return {
            "bool": {
                "should": [
                    TextSearchTemplates.multi_match_cross_fields(query_text, boost=2.0),
                    TextSearchTemplates.phrase_search(query_text, slop=3, boost=1.0)
                ],
                "minimum_should_match": 1
            }
        }


# ==================== EXPORTS ====================

__all__ = [
    "TextSearchTemplates",
    "validate_text_query_params",
    "optimize_text_search_for_performance", 
    "create_adaptive_text_search"
]