"""
🌐 API Package pour conversation service - Phase 1 (L0 Pattern Matching)

Ce module centralise les exports de l'API conversation service Phase 1
pour garantir la compatibilité avec le système de chargement
dynamique des routers dans local_app.py et heroku_app.py.

Pattern standardisé identique à search_service avec adaptations Phase 1.

✅ EXPORTS PHASE 1:
- router: FastAPI router avec endpoints L0
- initialize_pattern_matcher: Fonction d'initialisation Pattern Matcher
- get_pattern_matcher_direct: Accès direct au Pattern Matcher
"""

# ==========================================
# IMPORTS PHASE 1
# ==========================================

from .routes import (
    router, 
    initialize_pattern_matcher, 
    get_pattern_matcher_direct
)

# ==========================================
# EXPORTS PRINCIPAUX
# ==========================================

# Export explicite pour load_service_router() et initialisation
__all__ = [
    "router", 
    "initialize_pattern_matcher", 
    "get_pattern_matcher_direct",
    "API_ENDPOINTS",
    "ROUTER_INFO"
]

# ==========================================
# MÉTADONNÉES DU MODULE API
# ==========================================

__version__ = "1.0.0-phase1"
__description__ = "API REST pour conversation service Phase 1 - Pattern Matching L0"
__author__ = "Harena Finance Platform"
__phase__ = "L0_PATTERN_MATCHING"

# ==========================================
# CONFIGURATION DES ENDPOINTS PHASE 1
# ==========================================

API_ENDPOINTS = {
    # ===== ENDPOINTS PRINCIPAUX =====
    "chat": {
        "path": "/chat",
        "method": "POST",
        "description": "Classification d'intentions avec Pattern Matcher L0 (<10ms)",
        "response_model": "ChatResponse",
        "phase": "L0_PATTERN",
        "target_latency_ms": 10,
        "success_rate_target": 0.85
    },
    
    "health": {
        "path": "/health", 
        "method": "GET",
        "description": "Health check spécialisé Phase 1 avec métriques L0",
        "response_model": "ServiceHealth",
        "phase": "L0_PATTERN",
        "timeout_ms": 100
    },
    
    "metrics": {
        "path": "/metrics",
        "method": "GET", 
        "description": "Métriques détaillées Pattern Matcher L0",
        "phase": "L0_PATTERN",
        "includes": ["l0_performance", "pattern_usage", "cache_performance"]
    },
    
    "status": {
        "path": "/status",
        "method": "GET",
        "description": "Status et informations service Phase 1",
        "phase": "L0_PATTERN",
        "includes": ["architecture", "roadmap", "targets"]
    },
    
    # ===== ENDPOINTS DEBUG =====
    "test_patterns": {
        "path": "/debug/test-patterns",
        "method": "POST",
        "description": "Test debug patterns L0 avec analyse détaillée",
        "phase": "L0_PATTERN",
        "purpose": "debug"
    },
    
    "benchmark_l0": {
        "path": "/debug/benchmark-l0",
        "method": "POST",
        "description": "Benchmark performance L0 avec requêtes test",
        "phase": "L0_PATTERN",
        "purpose": "debug"
    },
    
    "patterns_info": {
        "path": "/debug/patterns-info",
        "method": "GET",
        "description": "Informations détaillées sur patterns chargés",
        "phase": "L0_PATTERN",
        "purpose": "debug"
    },
    
    # ===== ENDPOINTS VALIDATION =====
    "validate_phase1": {
        "path": "/validate-phase1",
        "method": "GET",
        "description": "Validation complète Phase 1 - Prêt pour Phase 2?",
        "phase": "L0_PATTERN",
        "purpose": "validation",
        "validates": ["targets", "performance", "readiness"]
    },
    
    # ===== ENDPOINTS ADMIN =====
    "add_pattern": {
        "path": "/admin/add-pattern",
        "method": "POST",
        "description": "Ajout pattern dynamique (admin seulement)",
        "phase": "L0_PATTERN",
        "purpose": "admin",
        "security": "admin_only"
    }
}

# ==========================================
# INFORMATIONS POUR LE CHARGEMENT DYNAMIQUE
# ==========================================

ROUTER_INFO = {
    "name": "conversation_service",
    "version": __version__,
    "phase": __phase__,
    "description": __description__,
    "endpoints_count": len(API_ENDPOINTS),
    
    # Configuration routing
    "prefix": "/api/v1/conversation",  # Préfixe pour local_app.py
    "tags": ["conversation-phase1", "pattern-matching", "l0"],
    
    # Dépendances Phase 1
    "dependencies": [
        "conversation_service.intent_detection.pattern_matcher",
        "conversation_service.models.conversation_models",
        "conversation_service.utils.logging"
    ],
    
    # Fonction d'initialisation requise
    "initialization_function": "initialize_pattern_matcher",
    "initialization_description": "Lie le Pattern Matcher L0 aux routes",
    
    # Validation function
    "validation_function": "get_pattern_matcher_direct",
    "validation_description": "Vérifie que le Pattern Matcher est accessible",
    
    # Performance expectations Phase 1
    "performance_expectations": {
        "avg_latency_ms": 8.5,
        "success_rate": 0.85,
        "l0_usage_percent": 80.0,
        "cache_hit_rate": 0.15,
        "throughput_requests_per_second": 100
    },
    
    # Capacités Phase 1
    "capabilities": [
        "Pattern matching ultra-rapide (<10ms)",
        "60+ patterns financiers optimisés", 
        "Extraction entités automatique",
        "Cache intelligent requêtes",
        "Métriques temps réel",
        "Debug et monitoring avancés"
    ],
    
    # Limitations Phase 1
    "limitations": [
        "Couverture limitée aux patterns prédéfinis",
        "Pas de classification contextuelle",
        "Pas de fallback intelligent",
        "Pas de conversation multi-tours"
    ]
}

# ==========================================
# FONCTIONS UTILITAIRES API
# ==========================================

def get_api_info() -> dict:
    """Informations complètes de l'API pour découverte"""
    return {
        "api": {
            "version": __version__,
            "phase": __phase__,
            "description": __description__,
            "endpoints_count": len(API_ENDPOINTS)
        },
        "endpoints": API_ENDPOINTS,
        "router_info": ROUTER_INFO,
        "usage": {
            "initialization": "initialize_pattern_matcher(pattern_matcher_instance)",
            "validation": "get_pattern_matcher_direct()",
            "main_endpoint": "POST /api/v1/conversation/chat"
        }
    }

def get_endpoints_by_category() -> dict:
    """Endpoints groupés par catégorie"""
    categories = {
        "main": [],
        "debug": [],
        "validation": [],
        "admin": [],
        "system": []
    }
    
    for endpoint_name, endpoint_info in API_ENDPOINTS.items():
        purpose = endpoint_info.get("purpose", "main")
        if purpose in categories:
            categories[purpose].append({
                "name": endpoint_name,
                "path": endpoint_info["path"],
                "method": endpoint_info["method"],
                "description": endpoint_info["description"]
            })
        else:
            categories["main"].append({
                "name": endpoint_name,
                "path": endpoint_info["path"], 
                "method": endpoint_info["method"],
                "description": endpoint_info["description"]
            })
    
    return categories

def validate_api_compatibility() -> dict:
    """Validation compatibilité API avec le système"""
    try:
        compatibility = {
            "router_available": router is not None,
            "initialization_function_available": initialize_pattern_matcher is not None,
            "validation_function_available": get_pattern_matcher_direct is not None,
            "endpoints_valid": len(API_ENDPOINTS) > 0,
            "router_info_complete": all(key in ROUTER_INFO for key in [
                "name", "version", "phase", "prefix", "dependencies"
            ])
        }
        
        # Test des exports
        for export in __all__:
            if export not in globals():
                compatibility["exports_valid"] = False
                compatibility["missing_export"] = export
                break
        else:
            compatibility["exports_valid"] = True
        
        compatibility["overall_compatible"] = all([
            compatibility["router_available"],
            compatibility["initialization_function_available"],
            compatibility["validation_function_available"],
            compatibility["endpoints_valid"],
            compatibility["router_info_complete"],
            compatibility["exports_valid"]
        ])
        
        return compatibility
        
    except Exception as e:
        return {
            "overall_compatible": False,
            "error": str(e)
        }

def get_phase1_readiness() -> dict:
    """Status de préparation Phase 1 pour intégration"""
    try:
        # Test basic functionality
        readiness = {
            "api_compatible": validate_api_compatibility()["overall_compatible"],
            "router_loadable": router is not None,
            "initialization_ready": initialize_pattern_matcher is not None,
            "endpoints_documented": len(API_ENDPOINTS) >= 8,  # Au moins 8 endpoints
            "phase1_complete": True
        }
        
        # Test d'initialisation si possible
        try:
            # Ne pas appeler get_pattern_matcher_direct() car il peut ne pas être initialisé
            readiness["validation_function_callable"] = callable(get_pattern_matcher_direct)
        except:
            readiness["validation_function_callable"] = False
        
        readiness["overall_ready"] = all([
            readiness["api_compatible"],
            readiness["router_loadable"],
            readiness["initialization_ready"],
            readiness["endpoints_documented"]
        ])
        
        return readiness
        
    except Exception as e:
        return {
            "overall_ready": False,
            "error": str(e)
        }

# ==========================================
# EXPORTS COMPLÉMENTAIRES
# ==========================================

# Ajout des fonctions utilitaires aux exports
__all__.extend([
    "get_api_info",
    "get_endpoints_by_category",
    "validate_api_compatibility", 
    "get_phase1_readiness"
])

# ==========================================
# VALIDATION AU CHARGEMENT
# ==========================================

# Validation basique au chargement du module
try:
    _api_compatibility = validate_api_compatibility()
    if not _api_compatibility.get("overall_compatible", False):
        import warnings
        warnings.warn(f"Conversation Service API Phase 1: Problèmes compatibilité détectés", UserWarning)
except Exception as validation_error:
    import warnings
    warnings.warn(f"Conversation Service API Phase 1: Erreur validation - {validation_error}", UserWarning)