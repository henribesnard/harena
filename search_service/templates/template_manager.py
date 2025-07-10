"""
Template Manager - Search Service

Gestionnaire central des templates de requêtes avec cache intelligent,
validation automatique et métadonnées complètes.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from copy import deepcopy
import hashlib
import json

from .config import TEMPLATE_CONFIG, PREDEFINED_TEMPLATES
from .exceptions import (
    TemplateNotFoundError, TemplateValidationError, 
    TemplateRenderError, InvalidParametersError, CacheError
)
from .text_search import TextSearchTemplates
from .financial_templates import FinancialQueryTemplates
from .query_builder import QueryTemplateBuilder
from ..models.service_contracts import IntentType, QueryType

logger = logging.getLogger(__name__)


@dataclass
class QueryTemplateMetadata:
    """Métadonnées complètes d'un template de requête."""
    name: str
    version: str
    description: str
    intent_types: List[IntentType]
    supported_query_types: List[QueryType]
    required_params: List[str]
    optional_params: List[str]
    default_params: Dict[str, Any]
    examples: List[Dict[str, Any]]
    performance_notes: List[str]
    elasticsearch_versions: List[str]
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: List[str]


class TemplateCache:
    """Cache intelligent pour les templates avec statistiques."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache = {}
        self._access_times = {}
        self._hit_count = 0
        self._miss_count = 0
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupère un template du cache."""
        if key in self._cache:
            # Vérifier TTL
            access_time = self._access_times.get(key, 0)
            if datetime.now().timestamp() - access_time < self._ttl_seconds:
                self._hit_count += 1
                self._access_times[key] = datetime.now().timestamp()
                return deepcopy(self._cache[key])
            else:
                # Expirer l'entrée
                del self._cache[key]
                del self._access_times[key]
        
        self._miss_count += 1
        return None
    
    def put(self, key: str, value: Dict[str, Any]):
        """Ajoute un template au cache."""
        # Gérer la taille du cache
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        self._cache[key] = deepcopy(value)
        self._access_times[key] = datetime.now().timestamp()
    
    def _evict_oldest(self):
        """Supprime l'entrée la plus ancienne."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Vide le cache."""
        self._cache.clear()
        self._access_times.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_requests = self._hit_count + self._miss_count
        hit_ratio = self._hit_count / max(total_requests, 1)
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": hit_ratio,
            "ttl_seconds": self._ttl_seconds
        }


class QueryTemplateManager:
    """
    Gestionnaire central des templates de requêtes avec cache intelligent
    et validation automatique des templates générés.
    """
    
    def __init__(self, cache_enabled: bool = True, cache_size: int = 1000):
        """
        Initialise le gestionnaire avec configuration du cache.
        """
        self._templates = {}
        self._cache = TemplateCache(cache_size) if cache_enabled else None
        self._load_default_templates()
        
        logger.info(f"✅ QueryTemplateManager initialisé (cache: {cache_enabled})")
    
    def _load_default_templates(self):
        """Charge les templates par défaut avec leurs métadonnées."""
        
        # Template de recherche textuelle générale
        self._templates[IntentType.TEXT_SEARCH] = {
            "template_func": TextSearchTemplates.multi_match_best_fields,
            "metadata": QueryTemplateMetadata(
                name="text_search_multi_match",
                version="1.0.0",
                description="Recherche textuelle multi-champs optimisée BM25",
                intent_types=[IntentType.TEXT_SEARCH],
                supported_query_types=[QueryType.LEXICAL],
                required_params=["query_text"],
                optional_params=["fields", "fuzziness", "tie_breaker", "operator", "boost"],
                default_params={
                    "fuzziness": "AUTO",
                    "tie_breaker": 0.3,
                    "operator": "or",
                    "boost": 1.0
                },
                examples=[
                    {
                        "description": "Recherche simple",
                        "params": {"query_text": "restaurant"},
                        "use_case": "Rechercher toutes les transactions liées aux restaurants"
                    }
                ],
                performance_notes=[
                    "Optimisé pour BM25",
                    "Utilise tie_breaker pour combiner les scores",
                    "Fuzziness AUTO adaptatif"
                ],
                elasticsearch_versions=["7.x", "8.x"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system",
                tags=["text", "multi_match", "bm25"]
            )
        }
        
        # Template de recherche par marchand
        self._templates[IntentType.MERCHANT_SEARCH] = {
            "template_func": FinancialQueryTemplates.merchant_fuzzy_search,
            "metadata": QueryTemplateMetadata(
                name="merchant_fuzzy_search",
                version="1.0.0",
                description="Recherche floue par nom de marchand",
                intent_types=[IntentType.MERCHANT_SEARCH],
                supported_query_types=[QueryType.LEXICAL],
                required_params=["merchant_name", "user_id"],
                optional_params=["fuzziness", "minimum_should_match", "boost"],
                default_params={
                    "fuzziness": "AUTO:3,6",
                    "minimum_should_match": "75%",
                    "boost": 1.0
                },
                examples=[
                    {
                        "description": "Recherche McDonald's",
                        "params": {"merchant_name": "McDonald's", "user_id": 12345},
                        "use_case": "Trouver toutes les transactions McDonald's avec tolérance aux erreurs"
                    }
                ],
                performance_notes=[
                    "Utilise champs merchant optimisés",
                    "Fuzziness adaptée aux noms de marchands",
                    "Boost élevé sur correspondances exactes"
                ],
                elasticsearch_versions=["7.x", "8.x"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system",
                tags=["merchant", "fuzzy", "financial"]
            )
        }
        
        # Template d'analyse des dépenses
        self._templates[IntentType.SPENDING_ANALYSIS] = {
            "template_func": FinancialQueryTemplates.spending_analysis_template,
            "metadata": QueryTemplateMetadata(
                name="spending_analysis_comprehensive",
                version="1.0.0",
                description="Analyse complète des dépenses avec filtres multiples",
                intent_types=[IntentType.SPENDING_ANALYSIS],
                supported_query_types=[QueryType.LEXICAL],
                required_params=["user_id", "period_start", "period_end"],
                optional_params=[
                    "categories", "merchants", "min_amount", "max_amount",
                    "exclude_categories", "boost"
                ],
                default_params={
                    "boost": 1.0
                },
                examples=[
                    {
                        "description": "Analyse mensuelle",
                        "params": {
                            "user_id": 12345,
                            "period_start": datetime(2024, 1, 1),
                            "period_end": datetime(2024, 1, 31),
                            "min_amount": 10.0
                        },
                        "use_case": "Analyser les dépenses de janvier 2024 supérieures à 10€"
                    }
                ],
                performance_notes=[
                    "Optimisé pour grandes plages de dates",
                    "Filtres combinés efficacement",
                    "Tri par montant par défaut"
                ],
                elasticsearch_versions=["7.x", "8.x"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system",
                tags=["spending", "analysis", "financial", "complex"]
            )
        }
        
        # Template de recherche par catégorie
        self._templates[IntentType.CATEGORY_SEARCH] = {
            "template_func": FinancialQueryTemplates.category_search_by_name,
            "metadata": QueryTemplateMetadata(
                name="category_search",
                version="1.0.0",
                description="Recherche par nom de catégorie",
                intent_types=[IntentType.CATEGORY_SEARCH],
                supported_query_types=[QueryType.LEXICAL],
                required_params=["category_name", "user_id"],
                optional_params=["include_subcategories", "exact_match", "boost"],
                default_params={
                    "include_subcategories": True,
                    "exact_match": False,
                    "boost": 1.0
                },
                examples=[],
                performance_notes=[],
                elasticsearch_versions=["7.x", "8.x"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system",
                tags=["category", "financial"]
            )
        }
        
        # Template de transactions récentes
        self._templates[IntentType.RECENT_TRANSACTIONS] = {
            "template_func": FinancialQueryTemplates.recent_transactions_template,
            "metadata": QueryTemplateMetadata(
                name="recent_transactions",
                version="1.0.0",
                description="Récupération des transactions récentes",
                intent_types=[IntentType.RECENT_TRANSACTIONS],
                supported_query_types=[QueryType.LEXICAL],
                required_params=["user_id"],
                optional_params=["days_back", "limit_amount", "boost"],
                default_params={
                    "days_back": 30,
                    "boost": 1.0
                },
                examples=[],
                performance_notes=[],
                elasticsearch_versions=["7.x", "8.x"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system",
                tags=["recent", "financial"]
            )
        }
    
    def get_template(
        self,
        intent_type: IntentType,
        use_cache: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Récupère et génère un template selon l'intention avec cache.
        """
        # Créer la clé de cache
        cache_key = self._create_cache_key(intent_type, params) if use_cache else None
        
        # Vérifier le cache
        if use_cache and self._cache and cache_key:
            cached_result = self._cache.get(cache_key)
            if cached_result:
                logger.debug(f"Template trouvé en cache: {intent_type}")
                return cached_result
        
        # Vérifier l'existence du template
        if intent_type not in self._templates:
            available = list(self._templates.keys())
            raise TemplateNotFoundError(str(intent_type), [str(t) for t in available])
        
        template_info = self._templates[intent_type]
        template_func = template_info["template_func"]
        metadata = template_info["metadata"]
        
        # Valider les paramètres requis
        missing_params = set(metadata.required_params) - set(params.keys())
        if missing_params:
            raise InvalidParametersError(missing_params=list(missing_params))
        
        # Fusionner avec les paramètres par défaut
        final_params = {**metadata.default_params, **params}
        
        # Générer la requête
        try:
            query = template_func(**final_params)
            
            # Valider la requête générée
            if not self._validate_generated_query(query):
                raise TemplateValidationError(f"Requête générée invalide pour {intent_type}")
            
            # Mettre en cache si activé
            if use_cache and self._cache and cache_key:
                self._cache.put(cache_key, query)
            
            logger.debug(f"Template généré avec succès: {intent_type}")
            return query
            
        except Exception as e:
            logger.error(f"Erreur génération template {intent_type}: {e}")
            raise TemplateRenderError(f"Impossible de générer le template: {e}")
    
    def create_builder(self, user_id: Optional[int] = None) -> QueryTemplateBuilder:
        """Crée un nouveau builder de requête."""
        return QueryTemplateBuilder(user_id=user_id)
    
    def get_template_with_builder(
        self,
        intent_type: IntentType,
        user_id: int,
        **params
    ) -> QueryTemplateBuilder:
        """
        Crée un builder pré-configuré selon l'intention.
        """
        builder = self.create_builder(user_id)
        builder.optimize_for_intent(intent_type)
        
        # Configuration spécifique par intention
        if intent_type == IntentType.TEXT_SEARCH and "query_text" in params:
            builder.add_text_search(params["query_text"])
            
        elif intent_type == IntentType.MERCHANT_SEARCH and "merchant_name" in params:
            builder.add_merchant_search(
                params["merchant_name"],
                exact_match=params.get("exact_match", False)
            )
            
        elif intent_type == IntentType.CATEGORY_SEARCH:
            if "category_ids" in params:
                builder.add_category_filter(category_ids=params["category_ids"])
            if "category_names" in params:
                builder.add_category_filter(category_names=params["category_names"])
        
        elif intent_type == IntentType.SPENDING_ANALYSIS:
            if "min_amount" in params or "max_amount" in params:
                builder.add_amount_range(
                    min_amount=params.get("min_amount"),
                    max_amount=params.get("max_amount")
                )
            if "period_start" in params or "period_end" in params:
                builder.add_date_range(
                    start_date=params.get("period_start"),
                    end_date=params.get("period_end")
                )
        
        return builder
    
    def get_predefined_template(
        self,
        template_name: str,
        params: Dict[str, Any],
        render: bool = True
    ) -> Dict[str, Any]:
        """
        Récupère un template prédéfini et le rend avec les paramètres.
        """
        if template_name not in PREDEFINED_TEMPLATES:
            available = list(PREDEFINED_TEMPLATES.keys())
            raise TemplateNotFoundError(template_name, available)
        
        template = deepcopy(PREDEFINED_TEMPLATES[template_name])
        
        if render:
            template = self._render_template(template, params)
        
        return template
    
    def get_available_templates(self) -> Dict[IntentType, QueryTemplateMetadata]:
        """Retourne les métadonnées de tous les templates disponibles."""
        return {
            intent_type: template_info["metadata"]
            for intent_type, template_info in self._templates.items()
        }
    
    def get_template_metadata(self, intent_type: IntentType) -> QueryTemplateMetadata:
        """Retourne les métadonnées d'un template spécifique."""
        if intent_type not in self._templates:
            raise TemplateNotFoundError(str(intent_type))
        
        return self._templates[intent_type]["metadata"]
    
    def register_custom_template(
        self,
        intent_type: IntentType,
        template_func: Callable,
        metadata: QueryTemplateMetadata
    ):
        """Enregistre un template personnalisé."""
        self._templates[intent_type] = {
            "template_func": template_func,
            "metadata": metadata
        }
        logger.info(f"Template personnalisé enregistré: {intent_type}")
    
    def update_template_metadata(
        self,
        intent_type: IntentType,
        **metadata_updates
    ):
        """Met à jour les métadonnées d'un template."""
        if intent_type not in self._templates:
            raise TemplateNotFoundError(str(intent_type))
        
        metadata = self._templates[intent_type]["metadata"]
        for key, value in metadata_updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now()
        logger.info(f"Métadonnées mises à jour pour {intent_type}")
    
    def clear_cache(self):
        """Vide le cache des templates."""
        if self._cache:
            self._cache.clear()
            logger.info("🗑️ Cache des templates vidé")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        if not self._cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            **self._cache.get_stats()
        }
    
    def validate_template_params(
        self,
        intent_type: IntentType,
        params: Dict[str, Any]
    ) -> bool:
        """Valide les paramètres pour un template donné."""
        if intent_type not in self._templates:
            raise TemplateNotFoundError(str(intent_type))
        
        metadata = self._templates[intent_type]["metadata"]
        
        # Vérifier les paramètres requis
        missing_params = set(metadata.required_params) - set(params.keys())
        if missing_params:
            raise InvalidParametersError(missing_params=list(missing_params))
        
        # Vérifier les paramètres invalides
        all_valid_params = set(metadata.required_params + metadata.optional_params)
        invalid_params = set(params.keys()) - all_valid_params
        if invalid_params:
            raise InvalidParametersError(invalid_params=list(invalid_params))
        
        return True
    
    def _create_cache_key(self, intent_type: IntentType, params: Dict[str, Any]) -> str:
        """Crée une clé de cache unique."""
        try:
            # Créer une représentation stable des paramètres
            sorted_params = json.dumps(params, sort_keys=True, default=str)
            
            # Créer le hash
            key_data = f"{intent_type.value}_{sorted_params}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Impossible de créer la clé de cache: {e}")
            return f"{intent_type.value}_{hash(frozenset(params.items()))}"
    
    def _validate_generated_query(self, query: Dict[str, Any]) -> bool:
        """Valide une requête générée."""
        try:
            # Vérifications de base
            if not isinstance(query, dict):
                return False
            
            # Vérifier la structure Elasticsearch valide
            valid_root_keys = {
                "match", "multi_match", "term", "terms", "range", "bool",
                "fuzzy", "wildcard", "prefix", "regexp", "exists", "match_all",
                "function_score", "dis_max", "constant_score", "nested"
            }
            
            query_keys = set(query.keys())
            if not query_keys.intersection(valid_root_keys):
                return False
            
            # Validation spécifique pour bool queries
            if "bool" in query:
                bool_query = query["bool"]
                if not isinstance(bool_query, dict):
                    return False
                
                valid_bool_keys = {"must", "should", "filter", "must_not", "minimum_should_match", "boost"}
                if not set(bool_query.keys()).issubset(valid_bool_keys):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation requête: {e}")
            return False
    
    def _render_template(
        self,
        template: Dict[str, Any],
        params: Dict[str, Any],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """Rend un template avec des paramètres."""
        try:
            rendered = deepcopy(template)
            
            def replace_placeholders(obj: Any, parameters: Dict[str, Any]) -> Any:
                """Remplace récursivement les placeholders."""
                if isinstance(obj, dict):
                    return {k: replace_placeholders(v, parameters) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_placeholders(item, parameters) for item in obj]
                elif isinstance(obj, str):
                    # Remplacer placeholders simples {{param}}
                    if obj.startswith("{{") and obj.endswith("}}"):
                        param_name = obj[2:-2].strip()
                        if param_name in parameters:
                            return parameters[param_name]
                        elif strict_mode:
                            raise ValueError(f"Paramètre manquant: {param_name}")
                        else:
                            return obj
                    
                    # Remplacer placeholders intégrés
                    import re
                    def replacer(match):
                        param_name = match.group(1).strip()
                        if param_name in parameters:
                            return str(parameters[param_name])
                        elif strict_mode:
                            raise ValueError(f"Paramètre manquant: {param_name}")
                        else:
                            return match.group(0)
                    
                    pattern = r'\{\{([^}]+)\}\}'
                    return re.sub(pattern, replacer, obj)
                else:
                    return obj
            
            return replace_placeholders(rendered, params)
            
        except Exception as e:
            logger.error(f"Erreur rendu template: {e}")
            raise TemplateRenderError(f"Impossible de rendre le template: {e}")


# ==================== FONCTIONS UTILITAIRES ====================

def create_template_from_intent(
    intent_type: IntentType,
    entities: Dict[str, Any],
    user_id: int,
    template_manager: Optional[QueryTemplateManager] = None
) -> Dict[str, Any]:
    """
    Crée automatiquement un template optimisé selon l'intention et les entités.
    """
    if template_manager is None:
        template_manager = QueryTemplateManager()
    
    # Mappage intention → paramètres
    params = {"user_id": user_id}
    
    # Extraction des entités selon l'intention
    if intent_type == IntentType.TEXT_SEARCH:
        params["query_text"] = entities.get("search_query", "")
        
    elif intent_type == IntentType.MERCHANT_SEARCH:
        params["merchant_name"] = entities.get("merchant_name", "")
        params["exact_match"] = entities.get("exact_match", False)
        
    elif intent_type == IntentType.SPENDING_ANALYSIS:
        params["period_start"] = entities.get("period_start")
        params["period_end"] = entities.get("period_end") 
        params["categories"] = entities.get("categories")
        params["min_amount"] = entities.get("min_amount")
        params["max_amount"] = entities.get("max_amount")
        
    elif intent_type == IntentType.CATEGORY_SEARCH:
        params["category_ids"] = entities.get("category_ids")
        params["category_names"] = entities.get("category_names")
    
    # Générer le template
    try:
        return template_manager.get_template(intent_type, **params)
    except Exception as e:
        logger.error(f"Erreur création template pour {intent_type}: {e}")
        # Fallback vers recherche générale
        return template_manager.get_template(
            IntentType.TEXT_SEARCH,
            query_text=entities.get("search_query", "*"),
            user_id=user_id
        )


def validate_query_template(template: Dict[str, Any]) -> bool:
    """
    Valide un template de requête Elasticsearch de manière exhaustive.
    """
    try:
        if not isinstance(template, dict):
            raise ValueError("Le template doit être un dictionnaire")
        
        # Types de requête supportés
        supported_query_types = {
            "match", "multi_match", "term", "terms", "range", "bool",
            "fuzzy", "wildcard", "prefix", "regexp", "exists", "match_all",
            "function_score", "dis_max", "constant_score", "nested", "has_child",
            "has_parent", "parent_id", "query_string", "simple_query_string",
            "geo_distance", "geo_bounding_box", "geo_polygon", "geo_shape",
            "more_like_this", "script", "script_score", "percolate"
        }
        
        def validate_query_structure(node: Dict[str, Any], path: str = "") -> bool:
            """Validation récursive de la structure de requête."""
            if not isinstance(node, dict):
                return True
            
            for key, value in node.items():
                current_path = f"{path}.{key}" if path else key
                
                # Validation des types de requête
                if key in supported_query_types:
                    if not isinstance(value, dict):
                        raise ValueError(f"Requête {key} invalide à {current_path}")
                    
                    # Validations spécifiques par type
                    if key == "multi_match":
                        if "query" not in value:
                            raise ValueError(f"multi_match manque 'query' à {current_path}")
                        if "fields" not in value:
                            raise ValueError(f"multi_match manque 'fields' à {current_path}")
                    
                    elif key == "bool":
                        valid_bool_keys = {"must", "should", "filter", "must_not", "minimum_should_match", "boost"}
                        invalid_keys = set(value.keys()) - valid_bool_keys
                        if invalid_keys:
                            raise ValueError(f"Clés bool invalides à {current_path}: {invalid_keys}")
                        
                        # Validation des clauses bool
                        for bool_clause in ["must", "should", "filter", "must_not"]:
                            if bool_clause in value:
                                if not isinstance(value[bool_clause], list):
                                    raise ValueError(f"{bool_clause} doit être une liste à {current_path}")
                                for i, clause in enumerate(value[bool_clause]):
                                    validate_query_structure(clause, f"{current_path}.{bool_clause}[{i}]")
                    
                    elif key == "range":
                        if not any(range_key in value for range_key in ["gte", "gt", "lte", "lt"]):
                            raise ValueError(f"Range manque opérateurs à {current_path}")
                    
                    elif key == "terms":
                        if not isinstance(value, dict) or len(value) != 1:
                            raise ValueError(f"Terms invalide à {current_path}")
                        field_name, terms_list = next(iter(value.items()))
                        if not isinstance(terms_list, list):
                            raise ValueError(f"Terms doit être une liste à {current_path}.{field_name}")
                
                # Validation récursive
                elif isinstance(value, dict):
                    validate_query_structure(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            validate_query_structure(item, f"{current_path}[{i}]")
            
            return True
        
        return validate_query_structure(template)
        
    except Exception as e:
        logger.error(f"Validation template échouée: {e}")
        return False


def optimize_query_for_performance(
    query: Dict[str, Any],
    max_clauses: int = 1024,
    enable_caching: bool = True
) -> Dict[str, Any]:
    """
    Optimise une requête pour les performances.
    """
    optimized = deepcopy(query)
    
    # Compter les clauses dans une requête bool
    def count_clauses(q: Dict[str, Any]) -> int:
        count = 0
        if isinstance(q, dict):
            if "bool" in q:
                bool_query = q["bool"]
                for clause_type in ["must", "should", "filter", "must_not"]:
                    if clause_type in bool_query:
                        count += len(bool_query[clause_type])
                        for clause in bool_query[clause_type]:
                            count += count_clauses(clause)
            elif any(key in q for key in ["query", "queries"]):
                for key in ["query", "queries"]:
                    if key in q:
                        if isinstance(q[key], list):
                            for sub_q in q[key]:
                                count += count_clauses(sub_q)
                        else:
                            count += count_clauses(q[key])
        return count
    
    # Vérifier et limiter le nombre de clauses
    clause_count = count_clauses(optimized)
    if clause_count > max_clauses:
        logger.warning(f"Requête avec {clause_count} clauses (max: {max_clauses})")
    
    # Ajouter des options de performance si activé
    if enable_caching and "query" in optimized:
        # Marquer pour mise en cache (selon la configuration ES)
        if "bool" in optimized["query"]:
            optimized["query"]["bool"]["_cache"] = True
    
    return optimized


# ==================== EXPORTS ====================

__all__ = [
    # Classes principales
    "QueryTemplateManager",
    "QueryTemplateMetadata",
    "TemplateCache",
    
    # Fonctions utilitaires
    "create_template_from_intent",
    "validate_query_template",
    "optimize_query_for_performance"
]