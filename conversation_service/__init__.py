"""
🏠 Conversation Service - Module Principal

Service ultra-optimisé de détection d'intention financière basé sur :
- 95% règles heuristiques intelligentes (0-5ms)
- 5% fallback DeepSeek optionnel (si nécessaire)
- Cache Redis hybride avec fallback mémoire
- Métriques temps réel et analytics

Architecture modulaire inspirée du fichier original qui fonctionne parfaitement,
avec améliorations pour production et maintenabilité.

Usage:
    from conversation_service import app
    # ou
    from conversation_service.services.intent_detection import get_intent_service
    # ou  
    uvicorn conversation_service.main:app --reload --port 8001

Version: 2.0.0
Auteur: Système optimisé basé sur tinybert_service.py
"""

import logging
from typing import Dict, Any, Optional, List

# Configuration logging par défaut
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Métadonnées du package
__version__ = "2.0.0"
__title__ = "Conversation Service"
__description__ = "Service ultra-optimisé de détection d'intention financière"
__author__ = "Harena AI Team"
__license__ = "MIT"

# Status du service
__status__ = "Production Ready"
__performance_target__ = "< 50ms latency, > 85% accuracy"


# =====================================
# IMPORTS PRINCIPAUX
# =====================================

try:
    # Import de l'application FastAPI principale
    from conversation_service.main import app
    
    # Services principaux
    from conversation_service.services.intent_detection.detector import (
        OptimizedIntentService, 
        get_intent_service_sync
    )
    from conversation_service.services.intent_detection.rule_engine import (
        IntelligentRuleEngine,
        get_rule_engine
    )
    
    # Modèles de données
    from conversation_service.models.intent import (
        IntentRequest, 
        IntentResponse, 
        BatchIntentRequest, 
        BatchIntentResponse,
        HealthResponse,
        MetricsResponse
    )
    from conversation_service.models.enums import (
        IntentType, 
        DetectionMethod, 
        ConfidenceLevel,
        CacheStrategy
    )
    
    # Cache et métriques
    from conversation_service.clients.cache.memory_cache import (
        HybridIntelligentCache,
        get_memory_cache
    )
    from conversation_service.utils.monitoring.intent_metrics import (
        IntentMetricsCollector,
        get_metrics_collector
    )
    
    # Configuration
    from conversation_service.config import config, get_supported_intents
    
    # Marquer imports comme réussis
    _imports_successful = True
    logger.info("✅ Module conversation_service chargé avec succès")
    
except ImportError as e:
    _imports_successful = False
    logger.error(f"❌ Erreur import module conversation_service: {e}")
    
    # Imports minimaux pour debugging
    app = None


# =====================================
# FONCTIONS UTILITAIRES PUBLIQUES
# =====================================

def get_service_info() -> Dict[str, Any]:
    """
    Informations complètes sur le service
    
    Returns:
        Dict avec métadonnées, statut et capacités
    """
    info = {
        "service": {
            "name": __title__,
            "version": __version__,
            "description": __description__,
            "status": __status__,
            "performance_target": __performance_target__,
            "author": __author__
        },
        "imports_successful": _imports_successful,
        "capabilities": {
            "rule_based_detection": True,
            "deepseek_fallback": True,
            "redis_cache": True,
            "batch_processing": True,
            "real_time_metrics": True,
            "entity_extraction": True
        },
        "supported_intents": len(get_supported_intents()) if _imports_successful else 0,
        "architecture": {
            "detection_methods": ["rules", "llm_fallback", "hybrid"],
            "cache_strategy": "redis_with_memory_fallback",
            "entity_extraction": "pattern_matching_with_contextual",
            "preprocessing": "french_financial_optimized"
        }
    }
    
    if _imports_successful:
        try:
            # Ajout infos configuration si disponible
            info["configuration"] = {
                "deepseek_enabled": config.service.enable_deepseek_fallback,
                "cache_enabled": config.rule_engine.enable_cache,
                "target_latency_ms": config.performance.target_latency_ms,
                "target_accuracy": config.performance.target_accuracy
            }
        except Exception as e:
            logger.warning(f"Erreur récupération configuration: {e}")
    
    return info


def quick_intent_detection(query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Détection rapide d'intention - Interface simplifiée
    
    Args:
        query: Texte utilisateur à analyser
        user_id: ID utilisateur optionnel
        
    Returns:
        Dict avec intention, confiance et entités
        
    Raises:
        RuntimeError: Si service non initialisé
    """
    if not _imports_successful:
        raise RuntimeError("Service non initialisé - erreurs import")
    
    try:
        # Utilisation service synchrone pour simplicité
        service = get_intent_service_sync()
        
        # Création requête
        request = IntentRequest(query=query, user_id=user_id)
        
        # Détection (note: fonction sync wrapper pour async)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(service.detect_intent(request))
        except RuntimeError:
            # Créer nouveau loop si nécessaire
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(service.detect_intent(request))
        
        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "entities": result["entities"],
            "method_used": result["method_used"],
            "processing_time_ms": result["processing_time_ms"]
        }
        
    except Exception as e:
        logger.error(f"Erreur détection rapide: {e}")
        return {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "entities": {},
            "method_used": "error",
            "processing_time_ms": 0.0,
            "error": str(e)
        }


def get_health_status() -> Dict[str, Any]:
    """
    Statut de santé du service
    
    Returns:
        Dict avec statut composants et métriques de base
    """
    if not _imports_successful:
        return {
            "status": "unhealthy",
            "reason": "import_errors",
            "imports_successful": False
        }
    
    try:
        service = get_intent_service_sync()
        
        # Health check asynchrone
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            health = loop.run_until_complete(service.health_check())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health = loop.run_until_complete(service.health_check())
        
        return health
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "imports_successful": True,
            "service_error": True
        }


def list_supported_intents() -> List[str]:
    """
    Liste des intentions supportées
    
    Returns:
        Liste des intentions financières et conversationnelles
    """
    if not _imports_successful:
        return []
    
    try:
        supported = get_supported_intents()
        return list(supported.keys())
    except Exception as e:
        logger.error(f"Erreur récupération intentions: {e}")
        return []


def validate_service() -> Dict[str, Any]:
    """
    Validation complète du service - Test de démarrage
    
    Returns:
        Dict avec résultats validation et recommandations
    """
    validation_results = {
        "overall_status": "unknown",
        "checks": {},
        "recommendations": [],
        "critical_issues": [],
        "warnings": []
    }
    
    # Check 1: Imports
    validation_results["checks"]["imports"] = _imports_successful
    if not _imports_successful:
        validation_results["critical_issues"].append("Imports failed - check dependencies")
        validation_results["overall_status"] = "failed"
        return validation_results
    
    # Check 2: Configuration
    try:
        config_valid = hasattr(config, 'service') and hasattr(config, 'performance')
        validation_results["checks"]["configuration"] = config_valid
        if not config_valid:
            validation_results["critical_issues"].append("Configuration invalid")
    except Exception:
        validation_results["checks"]["configuration"] = False
        validation_results["critical_issues"].append("Configuration error")
    
    # Check 3: Service initialization
    try:
        service = get_intent_service_sync()
        validation_results["checks"]["service_init"] = True
        
        # Test simple detection
        test_result = quick_intent_detection("bonjour")
        validation_results["checks"]["detection_test"] = test_result["intent"] == "GREETING"
        
        if test_result["intent"] != "GREETING":
            validation_results["warnings"].append("Basic detection test failed")
            
    except Exception as e:
        validation_results["checks"]["service_init"] = False
        validation_results["critical_issues"].append(f"Service initialization failed: {e}")
    
    # Check 4: Cache
    try:
        cache = get_memory_cache()
        cache_status = cache.get_redis_connection_status()
        validation_results["checks"]["cache"] = True
        validation_results["checks"]["redis_connection"] = cache_status.get("connected", False)
        
        if not cache_status.get("connected", False):
            validation_results["warnings"].append("Redis not connected - using memory fallback")
            
    except Exception:
        validation_results["checks"]["cache"] = False
        validation_results["warnings"].append("Cache initialization failed")
    
    # Détermination statut global
    if validation_results["critical_issues"]:
        validation_results["overall_status"] = "failed"
        validation_results["recommendations"].append("Fix critical issues before deployment")
    elif validation_results["warnings"]:
        validation_results["overall_status"] = "degraded"
        validation_results["recommendations"].append("Service functional but check warnings")
    else:
        validation_results["overall_status"] = "healthy"
        validation_results["recommendations"].append("Service ready for production")
    
    return validation_results


# =====================================
# EXPORTS PUBLICS
# =====================================

# Application FastAPI
__all__ = [
    # Application principale
    "app",
    
    # Services
    "OptimizedIntentService",
    "get_intent_service_sync", 
    "IntelligentRuleEngine",
    "get_rule_engine",
    
    # Modèles
    "IntentRequest",
    "IntentResponse", 
    "BatchIntentRequest",
    "BatchIntentResponse",
    "HealthResponse",
    "MetricsResponse",
    
    # Énumérations
    "IntentType",
    "DetectionMethod",
    "ConfidenceLevel", 
    "CacheStrategy",
    
    # Cache et métriques
    "HybridIntelligentCache",
    "get_memory_cache",
    "IntentMetricsCollector",
    "get_metrics_collector",
    
    # Configuration
    "config",
    "get_supported_intents",
    
    # Fonctions utilitaires
    "get_service_info",
    "quick_intent_detection",
    "get_health_status",
    "list_supported_intents",
    "validate_service",
    
    # Métadonnées
    "__version__",
    "__title__",
    "__description__"
]


# Message de bienvenue au chargement
if _imports_successful:
    logger.info(f"🎯 {__title__} v{__version__} - {__status__}")
    logger.info(f"🚀 Performance target: {__performance_target__}")
    logger.info("📋 Use 'from conversation_service import app' pour FastAPI")
    logger.info("🔧 Use 'quick_intent_detection(query)' pour test rapide")
else:
    logger.error("❌ Service non opérationnel - vérifiez les dépendances")


# Banner informatif (si module chargé directement)
if __name__ == "__main__":
    print("=" * 60)
    print(f"🎯 {__title__} v{__version__}")
    print(f"📋 {__description__}")
    print("=" * 60)
    
    # Affichage informations service
    info = get_service_info()
    print(f"✅ Imports: {'OK' if info['imports_successful'] else 'FAILED'}")
    print(f"🎯 Intentions: {info['supported_intents']}")
    print(f"🚀 Architecture: {info['architecture']['detection_methods']}")
    
    # Test validation
    print("\n🧪 Validation du service:")
    validation = validate_service()
    print(f"📊 Statut: {validation['overall_status']}")
    
    if validation['critical_issues']:
        print("❌ Problèmes critiques:")
        for issue in validation['critical_issues']:
            print(f"   - {issue}")
    
    if validation['warnings']:
        print("⚠️ Avertissements:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    print("\n🚀 Démarrage:")
    print("   uvicorn conversation_service.main:app --reload --port 8001")
    print("   ou")
    print("   python -m conversation_service.main")