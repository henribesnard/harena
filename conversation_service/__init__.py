"""
📦 Conversation Service Package - Phase 1 (L0 Pattern Matching)

Service de conversation financière avec Pattern Matcher L0 ultra-rapide
pour des performances <10ms sur 85%+ des requêtes financières courantes.

Architecture Phase 1:
- ✅ L0: Pattern Matching (<10ms, 85%+ hit rate)
- ❌ L1: TinyBERT Classification (Phase 2)
- ❌ L2: DeepSeek LLM Fallback (Phase 3)

Version: 1.0.0-phase1
Auteur: Harena Finance Platform
"""

# ==========================================
# IMPORTS PRINCIPAUX PHASE 1
# ==========================================

from .main import (
    app, 
    get_pattern_matcher, 
    is_service_ready, 
    get_service_phase,
    get_service_uptime,
    get_service_info
)

# ==========================================
# EXPORTS PRINCIPAUX
# ==========================================

# Export principal pour le chargement par local_app.py/heroku_app.py
__all__ = [
    "app", 
    "get_pattern_matcher", 
    "is_service_ready",
    "get_service_phase",
    "get_service_uptime", 
    "get_service_info",
    "SERVICE_INFO"
]

# ==========================================
# MÉTADONNÉES DU PACKAGE
# ==========================================

__version__ = "1.0.0-phase1"
__title__ = "Conversation Service - Phase 1"
__description__ = "Service de conversation financière avec Pattern Matching L0 ultra-rapide"
__author__ = "Harena Finance Platform"
__phase__ = "L0_PATTERN_MATCHING"

# ==========================================
# CONFIGURATION POUR LE CHARGEMENT DYNAMIQUE
# ==========================================

SERVICE_INFO = {
    "name": "conversation_service",
    "version": __version__,
    "phase": __phase__,
    "description": __description__,
    "app_module": "conversation_service.main",
    "app_factory": "app",
    "health_endpoint": "/health",
    "api_prefix": "/api/v1",
    "router_module": "conversation_service.api.routes",
    "router_prefix": "/api/v1/conversation",
    
    # Fonctions d'initialisation Phase 1
    "initialization_functions": {
        "pattern_matcher": "get_pattern_matcher",
        "service_ready": "is_service_ready",
        "service_info": "get_service_info"
    },
    
    # Dépendances Phase 1
    "dependencies": [
        "conversation_service.intent_detection.pattern_matcher",
        "conversation_service.models.conversation_models",
        "conversation_service.utils.logging"
    ],
    
    # Configuration requise Phase 1 (minimale)
    "required_config": [
        # Pas de config externe requise en Phase 1
    ],
    
    # Configuration optionnelle Phase 1
    "optional_config": [
        "CONVERSATION_SERVICE_DEBUG",
        "CONVERSATION_SERVICE_HOST",
        "CONVERSATION_SERVICE_PORT",
        "CONVERSATION_SERVICE_LOG_LEVEL"
    ],
    
    # Capabilities Phase 1
    "capabilities": [
        "Consultation soldes instantanée (<10ms)",
        "Virements simples avec extraction montants",
        "Gestion carte basique (blocage, activation)",
        "Analyse dépenses par catégorie",
        "Pattern matching 60+ patterns financiers",
        "Cache intelligent requêtes fréquentes",
        "Métriques temps réel et monitoring"
    ],
    
    # Limitations Phase 1
    "limitations": [
        "Pas de requêtes complexes multi-étapes",
        "Pas d'analyse contextuelle avancée",
        "Pas de conversations multi-tours",
        "Pas de conseil financier personnalisé",
        "Couverture limitée aux cas d'usage de base"
    ],
    
    # Targets de performance Phase 1
    "performance_targets": {
        "latency_ms": "<10",
        "success_rate": ">85%",
        "l0_usage_percent": ">80%",
        "cache_hit_rate": ">15%",
        "uptime": ">99.5%"
    },
    
    # Endpoints disponibles Phase 1
    "endpoints": {
        "main": {
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics", 
            "status": "/api/v1/status"
        },
        "debug": {
            "test_patterns": "/api/v1/debug/test-patterns",
            "benchmark_l0": "/api/v1/debug/benchmark-l0",
            "patterns_info": "/api/v1/debug/patterns-info"
        },
        "validation": {
            "phase1_ready": "/api/v1/validate-phase1"
        },
        "system": {
            "global_health": "/health",
            "readiness": "/health/ready",
            "liveness": "/health/live"
        }
    },
    
    # Roadmap future
    "roadmap": {
        "current_phase": "L0_PATTERN_MATCHING",
        "next_phase": "L1_LIGHTWEIGHT_CLASSIFIER",
        "future_phases": [
            "L2_DEEPSEEK_FALLBACK",
            "AUTOGEN_MULTI_AGENTS"
        ],
        "estimated_completion": {
            "phase1": "COMPLETED",
            "phase2": "2-3 weeks après validation Phase 1",
            "phase3": "4-6 weeks après Phase 2",
            "phase4": "8-12 weeks après Phase 3"
        }
    }
}

# ==========================================
# INFORMATIONS DEBUG ET MONITORING
# ==========================================

def get_package_info() -> dict:
    """Informations complètes du package pour monitoring"""
    return {
        "package": __title__,
        "version": __version__,
        "phase": __phase__,
        "description": __description__,
        "author": __author__,
        "service_info": SERVICE_INFO,
        "capabilities": SERVICE_INFO["capabilities"],
        "limitations": SERVICE_INFO["limitations"],
        "performance_targets": SERVICE_INFO["performance_targets"]
    }

def get_phase1_status() -> dict:
    """Status Phase 1 pour validation externe"""
    try:
        return {
            "phase": __phase__,
            "version": __version__,
            "service_ready": is_service_ready(),
            "uptime_seconds": get_service_uptime(),
            "service_phase": get_service_phase(),
            "detailed_info": get_service_info()
        }
    except Exception as e:
        return {
            "phase": __phase__,
            "version": __version__,
            "error": str(e),
            "service_ready": False
        }

# ==========================================
# VALIDATION COMPATIBILITÉ
# ==========================================

def validate_phase1_compatibility() -> dict:
    """Validation compatibilité avec le reste du système"""
    try:
        compatibility = {
            "app_available": app is not None,
            "pattern_matcher_available": False,
            "service_ready": False,
            "exports_valid": True
        }
        
        # Test pattern matcher
        try:
            matcher = get_pattern_matcher()
            compatibility["pattern_matcher_available"] = matcher is not None
        except:
            compatibility["pattern_matcher_available"] = False
        
        # Test service ready
        try:
            compatibility["service_ready"] = is_service_ready()
        except:
            compatibility["service_ready"] = False
        
        # Test exports
        for export in __all__:
            if export not in globals():
                compatibility["exports_valid"] = False
                compatibility[f"missing_export"] = export
                break
        
        compatibility["overall_compatible"] = all([
            compatibility["app_available"],
            compatibility["exports_valid"]
        ])
        
        return compatibility
        
    except Exception as e:
        return {
            "overall_compatible": False,
            "error": str(e)
        }

# ==========================================
# EXPORTS COMPLÉMENTAIRES
# ==========================================

# Ajout des fonctions utilitaires aux exports
__all__.extend([
    "get_package_info",
    "get_phase1_status", 
    "validate_phase1_compatibility"
])

# ==========================================
# VALIDATION AU CHARGEMENT
# ==========================================

# Validation basique au chargement du module
try:
    _compatibility = validate_phase1_compatibility()
    if not _compatibility.get("exports_valid", False):
        import warnings
        warnings.warn(f"Conversation Service Phase 1: Exports manquants détectés", UserWarning)
except Exception as validation_error:
    import warnings
    warnings.warn(f"Conversation Service Phase 1: Erreur validation - {validation_error}", UserWarning)