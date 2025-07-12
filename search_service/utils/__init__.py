"""
Module utils du Search Service - Utilitaires et services transversaux
====================================================================

Ce module regroupe tous les utilitaires essentiels du Search Service :
- Cache LRU haute performance  
- Validateurs de requêtes

Architecture :
    Core Components → Utils → External Services
    
Utilisé par :
    - Tous les modules core pour le cache
    - API routes pour la validation
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

# === IMPORTS RÉELS UNIQUEMENT ===

# Import du cache (✅ EXISTE)
from .cache import (
    # Classes de cache
    LRUCache,
    CacheEntry,
    CacheStats,
    CacheStrategy,
    CacheLevel,
    EvictionPolicy,
    
    # Gestionnaire de cache
    CacheManager,
    SmartCache,
    CacheKeyGenerator,
    
    # Instances globales  
    cache_manager,
    global_cache_manager,
    
    # Fonctions utilitaires
    create_cache_key,
    serialize_cache_value,
    deserialize_cache_value,
    cleanup_expired_entries,
    get_cache_statistics,
    get_cache_stats,
    clear_all_caches,
    periodic_cleanup,
    
    # Décorateurs
    cached
)

# Import des validateurs (✅ EXISTE - classes réelles uniquement)
from .validators import (
    # Exception
    ValidationError,
    
    # Classes de validation RÉELLES
    SecurityValidator,
    ContractValidator,
    FieldValidator,
    ElasticsearchQueryValidator,
    PerformanceValidator,
    BatchValidator,
    FinancialDataValidator,
    
    # Factory
    ValidatorFactory,
    
    # Fonctions utilitaires RÉELLES
    sanitize_query_string,
    is_valid_user_id,
    get_field_type,
    validate_query_timeout,
    estimate_result_size
)

# === CONFIGURATION DU LOGGING ===

logger = logging.getLogger(__name__)

# === ALIAS POUR COMPATIBILITÉ (si vraiment nécessaires) ===

# Créer des alias SEULEMENT si utilisés ailleurs
QueryValidator = ContractValidator
RequestValidator = ContractValidator
ResponseValidator = ContractValidator

# Types de validation simplifiés
class ValidationRule:
    """Type de base pour les règles de validation"""
    pass

class ValidationResult:
    """Résultat de validation simplifié"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.error_message = "; ".join(self.errors) if self.errors else None

# === FONCTIONS WRAPPER MINIMALES ===

def validate_search_request(request_data: Dict[str, Any]) -> ValidationResult:
    """Wrapper pour validation de requête"""
    try:
        is_valid, errors = ContractValidator.validate_search_query(request_data)
        return ValidationResult(is_valid, errors)
    except Exception as e:
        return ValidationResult(False, [str(e)])

def validate_elasticsearch_query(es_query: Dict[str, Any]) -> ValidationResult:
    """Wrapper pour validation Elasticsearch"""
    try:
        is_valid, errors = ElasticsearchQueryValidator.validate_query_body(es_query)
        return ValidationResult(is_valid, errors)
    except Exception as e:
        return ValidationResult(False, [str(e)])

def validate_user_permissions(user_id: int, operation: str) -> bool:
    """Validation simple des permissions utilisateur"""
    return is_valid_user_id(user_id)

def validate_date_ranges(start_date: str, end_date: str) -> bool:
    """Validation de plages de dates"""
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        return start <= end
    except Exception:
        return False

def validate_aggregation_request(agg_request: Dict[str, Any]) -> ValidationResult:
    """Validation des requêtes d'agrégation"""
    errors = []
    
    if not isinstance(agg_request, dict):
        errors.append("Aggregation request must be a dictionary")
    
    if "types" in agg_request and not isinstance(agg_request["types"], list):
        errors.append("Aggregation types must be a list")
    
    return ValidationResult(len(errors) == 0, errors)

def sanitize_search_input(text: str) -> str:
    """Alias pour sanitize_query_string"""
    return sanitize_query_string(text)

# === GESTIONNAIRE UTILS MINIMAL ===

class UtilsManager:
    """Gestionnaire centralisé des utilitaires du Search Service"""
    
    def __init__(self):
        self._initialized = False
        self._cache_manager = None
        
        logger.info("UtilsManager créé")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialise les utilitaires du Search Service"""
        if self._initialized:
            logger.warning("UtilsManager déjà initialisé")
            return {"status": "already_initialized"}
        
        try:
            logger.info("Initialisation des utilitaires du Search Service...")
            
            # Initialiser le gestionnaire de cache
            self._cache_manager = cache_manager
            
            self._initialized = True
            logger.info("✅ Utilitaires initialisés avec succès")
            
            return {
                "status": "success",
                "components": {
                    "cache": "initialized",
                    "validators": "initialized"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des utilitaires: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Arrêt propre de tous les utilitaires"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            if self._cache_manager:
                clear_all_caches()
            
            self._initialized = False
            logger.info("✅ Utilitaires arrêtés")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Vérification de santé des utilitaires"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            health_status = {
                "system_status": "healthy",
                "components": {
                    "cache": {"status": "healthy", "stats": get_cache_stats()},
                    "validators": {"status": "healthy"}
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé: {e}")
            return {"status": "error", "error": str(e)}
    
    @property
    def cache_manager(self):
        """Accès au gestionnaire de cache"""
        return self._cache_manager if self._initialized else None

# === INSTANCE GLOBALE ===

utils_manager = UtilsManager()

# === FONCTIONS D'INTERFACE PUBLIQUE ===

async def initialize_utils(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialise tous les utilitaires du Search Service"""
    return await utils_manager.initialize(config)

async def shutdown_utils() -> Dict[str, Any]:
    """Arrêt propre de tous les utilitaires"""
    return await utils_manager.shutdown()

async def get_utils_health() -> Dict[str, Any]:
    """Vérification de santé globale des utilitaires"""
    return await utils_manager.get_health_status()

def get_cache_manager():
    """Accès sécurisé au gestionnaire de cache"""
    return utils_manager.cache_manager

# === FONCTIONS UTILITAIRES INTÉGRÉES ===

def validate_and_cache_query(query_data: Dict[str, Any], 
                            cache_key: str,
                            ttl_seconds: int = 300) -> Tuple[bool, Optional[str], Any]:
    """Valide une requête et vérifie le cache"""
    try:
        # Validation
        validation_result = validate_search_request(query_data)
        if not validation_result.is_valid:
            return False, validation_result.error_message, None
        
        # Vérification cache
        if utils_manager.cache_manager:
            search_cache = utils_manager.cache_manager.get_cache("search")
            if search_cache:
                result = search_cache.get(cache_key)
                if result is not None:
                    return True, None, result
        
        return True, None, None
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation/cache: {e}")
        return False, str(e), None

def cache_search_result(cache_key: str, result: Any, ttl_seconds: int = 300):
    """Met en cache un résultat de recherche"""
    if utils_manager.cache_manager:
        try:
            search_cache = utils_manager.cache_manager.get_cache("search")
            if search_cache:
                search_cache.set(cache_key, result, ttl_seconds)
        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache: {e}")

def get_utils_status() -> Dict[str, str]:
    """Statut simple des utilitaires pour les health checks"""
    if not utils_manager._initialized:
        return {"status": "not_ready", "reason": "utils_not_initialized"}
    
    try:
        if utils_manager._cache_manager is not None:
            return {"status": "ready"}
        else:
            return {"status": "partial", "reason": "cache_missing"}
            
    except Exception as e:
        return {"status": "error", "reason": str(e)}

# === EXPORTS PRINCIPAUX (SEULEMENT CE QUI EXISTE) ===

__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "UtilsManager",
    "utils_manager",
    
    # === FONCTIONS D'INITIALISATION ===
    "initialize_utils",
    "shutdown_utils",
    "get_utils_health",
    "get_cache_manager",
    
    # === FONCTIONS INTÉGRÉES ===
    "validate_and_cache_query",
    "cache_search_result",
    "get_utils_status",
    
    # === CACHE (✅ EXISTE) ===
    "LRUCache",
    "CacheEntry",
    "CacheStats", 
    "CacheStrategy",
    "CacheLevel",
    "EvictionPolicy",
    "CacheManager",
    "SmartCache",
    "CacheKeyGenerator",
    "cache_manager",
    "global_cache_manager",
    "create_cache_key",
    "serialize_cache_value",
    "deserialize_cache_value",
    "cleanup_expired_entries",
    "get_cache_statistics",
    "get_cache_stats",
    "clear_all_caches",
    "periodic_cleanup",
    "cached",
    
    # === VALIDATEURS (✅ EXISTE) ===
    "ValidationError",
    "SecurityValidator",
    "ContractValidator",
    "FieldValidator", 
    "ElasticsearchQueryValidator",
    "PerformanceValidator",
    "BatchValidator",
    "FinancialDataValidator",
    "ValidatorFactory",
    "sanitize_query_string",
    "is_valid_user_id",
    "get_field_type",
    "validate_query_timeout",
    "estimate_result_size",
    
    # === ALIAS COMPATIBILITÉ (si vraiment nécessaires) ===
    "QueryValidator",
    "RequestValidator", 
    "ResponseValidator",
    "ValidationRule",
    "ValidationResult",
    "validate_search_request",
    "validate_elasticsearch_query",
    "validate_user_permissions",
    "validate_date_ranges",
    "validate_aggregation_request",
    "sanitize_search_input",
]

# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team" 
__description__ = "Utilitaires et services transversaux du Search Service"

# Logging de l'import du module
logger.info(f"Module utils initialisé - version {__version__}")
logger.info("Composants disponibles: Cache=✅, Validators=✅")