"""
🚀 Point d'Entrée Principal - Service Détection d'Intention

Application FastAPI complète reprenant la logique éprouvée du fichier original
avec architecture modulaire et optimisations production.

Démarrage: uvicorn conversation_service.main:app --reload --port 8001
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from conversation_service.config import config
from conversation_service.services.intent_detection.detector import get_intent_service_sync
from conversation_service.api.routes.intent import router as intent_router
from conversation_service.utils.monitoring.intent_metrics import get_metrics_collector
from conversation_service.models.exceptions import ConversationServiceError

# Configuration logging
logging.basicConfig(
    level=getattr(logging, config.service.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire cycle de vie application
    Reprend la logique du fichier original avec améliorations
    """
    startup_time = time.time()
    
    print("🚀 Démarrage Service Ultra-Optimisé")
    logger.info("Initialisation service de détection d'intention...")
    
    try:
        # Initialisation service principal (asynchrone)
        intent_service = get_intent_service_sync()
        await intent_service.initialize()
        
        # Vérification santé initiale
        health_status = await intent_service.health_check()
        
        if health_status["status"] == "healthy":
            print("✅ Service prêt - Règles intelligentes activées")
            logger.info("Service démarré avec succès")
        elif health_status["status"] == "degraded":
            print("⚠️ Service démarré en mode dégradé (DeepSeek indisponible)")
            logger.warning("Service en mode dégradé - fallback LLM indisponible")
        else:
            print("❌ Service démarré avec erreurs")
            logger.error(f"Service en état critique: {health_status}")
        
        # Affichage configuration
        print(f"📊 Configuration active:")
        print(f"   - DeepSeek fallback: {'✅' if config.service.enable_deepseek_fallback else '❌'}")
        print(f"   - Cache Redis: {'✅' if hasattr(intent_service, 'cache_manager') else '❌'}")
        print(f"   - Cible latence: {config.performance.target_latency_ms}ms")
        print(f"   - Cible précision: {config.performance.target_accuracy}")
        
        startup_duration = (time.time() - startup_time) * 1000
        print(f"🎯 Service initialisé en {startup_duration:.1f}ms")
        
        # Collecte métriques de démarrage
        metrics_collector = get_metrics_collector()
        logger.info("Collecteur de métriques activé")
        
        yield
        
    except Exception as e:
        logger.error(f"Erreur initialisation service: {e}")
        print(f"❌ Échec initialisation: {e}")
        raise
    
    finally:
        # Nettoyage à l'arrêt
        print("🔄 Arrêt du service...")
        logger.info("Service en cours d'arrêt")
        
        try:
            # Sauvegarde métriques finales
            final_metrics = intent_service.get_metrics()
            logger.info(
                f"Service arrêté - Statistiques finales: "
                f"{final_metrics.get('total_requests', 0)} requêtes traitées, "
                f"latence moyenne: {final_metrics.get('avg_latency_ms', 0):.1f}ms"
            )
            print("✅ Service arrêté proprement")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")


# Création application FastAPI
app = FastAPI(
    title=config.service.service_name,
    description=config.service.description,
    version=config.service.version,
    lifespan=lifespan,
    docs_url="/docs" if config.service.enable_docs else None,
    redoc_url="/redoc" if config.service.enable_docs else None,
    openapi_url="/openapi.json" if config.service.enable_docs else None
)


# =====================================
# MIDDLEWARE CONFIGURATION
# =====================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.service.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"]
)

# Trusted Host Middleware (sécurité production)
if config.service.cors_origins != ["*"]:
    trusted_hosts = [origin.replace("https://", "").replace("http://", "") 
                    for origin in config.service.cors_origins]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


# =====================================
# MIDDLEWARE CUSTOM
# =====================================

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Middleware de mesure temps de traitement"""
    start_time = time.time()
    
    # Génération ID requête unique
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    # Ajout headers de contexte
    request.state.request_id = request_id
    request.state.start_time = start_time
    
    # Traitement requête
    try:
        response = await call_next(request)
        
        # Calcul temps de traitement
        processing_time = (time.time() - start_time) * 1000
        
        # Ajout headers de réponse
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.2f}ms"
        
        # Logging requête
        logger.debug(
            f"Request processed: {request.method} {request.url.path} "
            f"in {processing_time:.2f}ms (ID: {request_id})"
        )
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"after {processing_time:.2f}ms - {str(e)} (ID: {request_id})"
        )
        raise


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Middleware de gestion d'erreurs globale"""
    try:
        return await call_next(request)
        
    except ConversationServiceError as e:
        # Erreurs métier gérées
        logger.warning(f"Business error: {e.error_code} - {e.message}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
                "severity": str(e.severity),
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )
        
    except HTTPException:
        # HTTPException déjà gérées par FastAPI
        raise
        
    except Exception as e:
        # Erreurs non gérées
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Une erreur inattendue s'est produite",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )


# =====================================
# GESTIONNAIRES D'EXCEPTIONS
# =====================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gestionnaire d'erreurs de validation Pydantic"""
    
    # Extraction détails erreurs
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Données de requête invalides",
            "validation_errors": errors,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Gestionnaire 404 personnalisé"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "NOT_FOUND",
            "message": f"Endpoint {request.url.path} introuvable",
            "available_endpoints": [
                "/api/v1/detect-intent",
                "/api/v1/health", 
                "/api/v1/metrics",
                "/docs"
            ],
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# =====================================
# ROUTES PRINCIPALES
# =====================================

# Inclusion du router principal
app.include_router(intent_router)


# Routes racine et statut
@app.get("/")
async def root():
    """Endpoint racine avec informations service"""
    return {
        "service": config.service.service_name,
        "version": config.service.version,
        "description": config.service.description,
        "status": "operational",
        "endpoints": {
            "health": "/api/v1/health",
            "detect_intent": "/api/v1/detect-intent",
            "metrics": "/api/v1/metrics",
            "documentation": "/docs"
        },
        "features": {
            "rule_based_detection": True,
            "deepseek_fallback": config.service.enable_deepseek_fallback,
            "redis_cache": True,
            "batch_processing": config.service.enable_batch_processing,
            "metrics_collection": config.service.enable_metrics_collection
        }
    }


@app.get("/status")
async def status():
    """Endpoint statut rapide pour load balancers"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/version")
async def version():
    """Endpoint version pour déploiements"""
    return {
        "version": config.service.version,
        "service": config.service.service_name,
        "build_timestamp": time.time()
    }


# =====================================
# ENDPOINTS DEBUG (développement)
# =====================================

@app.get("/debug/config")
async def debug_config():
    """Configuration active (développement seulement)"""
    if config.service.log_level != "DEBUG":
        raise HTTPException(status_code=404, detail="Debug endpoints disabled")
    
    return {
        "service_config": {
            "name": config.service.service_name,
            "version": config.service.version,
            "log_level": config.service.log_level,
            "cors_origins": config.service.cors_origins
        },
        "performance_config": {
            "target_latency_ms": config.performance.target_latency_ms,
            "target_accuracy": config.performance.target_accuracy,
            "cache_max_size": config.performance.cache_max_size,
            "cache_ttl_seconds": config.performance.cache_ttl_seconds
        },
        "feature_flags": {
            "deepseek_enabled": config.service.enable_deepseek_fallback,
            "batch_processing": config.service.enable_batch_processing,
            "metrics_collection": config.service.enable_metrics_collection
        }
    }


# =====================================
# FONCTION PRINCIPALE
# =====================================

async def run_performance_test():
    """
    Test de performance reprenant la logique du fichier original
    Pour validation et benchmark du service
    """
    from conversation_service.services.intent_detection.detector import get_intent_service_sync
    from conversation_service.models.intent import IntentRequest
    
    service = get_intent_service_sync()
    await service.initialize()
    
    test_queries = [
        "bonjour comment allez vous",
        "quel est mon solde", 
        "mes dépenses restaurant ce mois",
        "virement 100 euros à Paul",
        "historique janvier",
        "bloquer ma carte",
        "au revoir",
        "aide moi s'il te plaît",
        "requête très ambiguë pour tester"
    ]
    
    print("🧪 Test Performance Service Ultra-Optimisé")
    print("=" * 80)
    
    for query in test_queries:
        request = IntentRequest(query=query, use_deepseek_fallback=True)
        result = await service.detect_intent(request)
        
        entities_str = ", ".join([f"{k}:{v}" for k, v in result["entities"].items()]) if result["entities"] else "none"
        
        print(f"{query[:35]:<35} | {result['intent']:<18} | {result['confidence']:.3f} | {result['processing_time_ms']:5.1f}ms | {result['method_used']:<15} | {entities_str}")
    
    print("\n📊 Métriques finales:")
    metrics = service.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    # Test direct (développement)
    print("🔧 Mode développement - Test direct")
    asyncio.run(run_performance_test())