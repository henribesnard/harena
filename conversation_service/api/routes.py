from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import time
import logging

from ..models.conversation import (
    ChatRequest, 
    ChatResponse, 
    HealthResponse, 
    MetricsResponse, 
    ConfigResponse,
    ProcessingMetadata,
    ProcessingError,
    ValidationError
)
from conversation_service.agents.intent_classifier import intent_classifier
from conversation_service.clients.deepseek_client import deepseek_client
from conversation_service.config.settings import settings

logger = logging.getLogger(__name__)

# ✅ ARCHITECTURE UNIQUE - Router principal seul
router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["conversation"])
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Endpoint principal de conversation - Classification d'intentions
    
    Args:
        request: Requête de conversation
        background_tasks: Tâches en arrière-plan
        
    Returns:
        ChatResponse: Réponse avec intention classifiée
    """
    start_time = time.time()
    
    try:
        logger.info(f"Nouvelle conversation - User: {request.user_id}, Message: '{request.message[:100]}...'")
        
        # Classification d'intention
        intent_result = await intent_classifier.classify_intent(request.message)
        
        # Génération de la réponse textuelle
        response_text = _generate_response_text(intent_result)
        
        # Calcul du temps de traitement
        processing_time = int((time.time() - start_time) * 1000)
        
        # Métadonnées de traitement
        metadata = ProcessingMetadata(
            processing_time_ms=processing_time,
            agent_used="intent_classifier",
            model_used=settings.DEEPSEEK_CHAT_MODEL,
            cache_hit=False  # TODO: Intégrer avec cache DeepSeek
        )
        
        # Construction de la réponse
        response = ChatResponse(
            response=response_text,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            entities=intent_result.entities,
            is_clear=intent_result.confidence >= settings.MIN_CONFIDENCE_THRESHOLD,
            clarification_needed=_get_clarification_message(intent_result) if intent_result.confidence < settings.MIN_CONFIDENCE_THRESHOLD else None,
            metadata=metadata
        )
        
        # Log de succès
        logger.info(f"Conversation réussie - Intent: {intent_result.intent.value}, Confiance: {intent_result.confidence:.2f}, Temps: {processing_time}ms")
        
        # Tâche en arrière-plan pour analytics (si nécessaire)
        background_tasks.add_task(_log_conversation_analytics, request, response)
        
        return response
        
    except ProcessingError as e:
        logger.error(f"Erreur de traitement: {e.message}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "processing_error",
                "message": e.message,
                "details": e.details
            }
        )
    except ValidationError as e:
        logger.error(f"Erreur de validation: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error", 
                "message": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Une erreur inattendue s'est produite",
                "details": {"error": str(e)}
            }
        )

@router.get("/health", response_model=HealthResponse, tags=["conversation_system"])
async def health_check() -> HealthResponse:
    """Endpoint de santé du service"""
    
    try:
        # Vérification DeepSeek
        deepseek_health = await deepseek_client.health_check()
        
        # Vérification agent
        agent_metrics = intent_classifier.get_metrics()
        
        # Statut global
        is_healthy = (
            deepseek_health["status"] == "healthy" and
            agent_metrics["total_classifications"] >= 0  # Agent opérationnel
        )
        
        dependencies = {
            "deepseek": deepseek_health["status"],
            "intent_classifier": "operational",
            "cache": "operational"
        }
        
        return HealthResponse(
            status="healthy" if is_healthy else "degraded",
            version=settings.API_VERSION,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.API_VERSION,
            dependencies={"error": str(e)}
        )

@router.get("/metrics", response_model=MetricsResponse, tags=["conversation_system"])
async def get_metrics() -> MetricsResponse:
    """Endpoint des métriques du service"""
    
    try:
        # Métriques agent
        agent_metrics = intent_classifier.get_metrics()
        
        # Métriques DeepSeek
        deepseek_metrics = deepseek_client.get_metrics()
        
        return MetricsResponse(
            total_classifications=agent_metrics["total_classifications"],
            success_rate=agent_metrics["success_rate"],
            avg_processing_time_ms=agent_metrics["avg_processing_time"] * 1000,
            avg_confidence=agent_metrics["avg_confidence"],
            intent_distribution=agent_metrics["intent_distribution"],
            cache_hit_rate=deepseek_metrics["cache_hit_rate"]
        )
        
    except Exception as e:
        logger.error(f"Erreur métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "metrics_error", "message": str(e)}
        )

@router.get("/config", response_model=ConfigResponse, tags=["conversation_system"])
async def get_config() -> ConfigResponse:
    """Endpoint de configuration du service"""
    
    try:
        # Configuration publique (sans secrets)
        public_config = {
            "min_confidence_threshold": settings.MIN_CONFIDENCE_THRESHOLD,
            "supported_intents": [intent.value for intent in intent_classifier._metrics.get("intent_distribution", {}).keys()],
            "cache_enabled": True,
            "cache_ttl_seconds": settings.CLASSIFICATION_CACHE_TTL,
            "model_used": settings.DEEPSEEK_CHAT_MODEL,
            "request_timeout_seconds": settings.REQUEST_TIMEOUT
        }
        
        return ConfigResponse(
            service_name=settings.API_TITLE,
            version=settings.API_VERSION,
            configuration=public_config
        )
        
    except Exception as e:
        logger.error(f"Erreur configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "config_error", "message": str(e)}
        )

@router.post("/clear-cache", tags=["conversation_system"])
async def clear_cache() -> Dict[str, Any]:
    """Endpoint pour vider le cache (développement/debug)"""
    
    try:
        # Clear cache DeepSeek
        deepseek_client.clear_cache()
        
        # Reset métriques agent si nécessaire
        # intent_classifier.reset_metrics()
        
        return {
            "status": "success",
            "message": "Cache vidé avec succès",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Erreur clear cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "cache_error", "message": str(e)}
        )

def _generate_response_text(intent_result) -> str:
    """Génère le texte de réponse basé sur l'intention"""
    
    intent_responses = {
        "search_by_merchant": "J'ai détecté une recherche par marchand. Je vais chercher vos transactions avec ce marchand.",
        "search_by_category": "J'ai détecté une recherche par catégorie. Je vais chercher vos transactions dans cette catégorie.",
        "search_by_amount": "J'ai détecté une recherche par montant. Je vais chercher vos transactions selon ce critère de montant.",
        "search_by_date": "J'ai détecté une recherche par date. Je vais chercher vos transactions pour cette période.",
        "search_general": "J'ai détecté une recherche générale. Je vais chercher dans toutes vos transactions.",
        "spending_analysis": "J'ai détecté une demande d'analyse de dépenses. Je vais analyser vos dépenses.",
        "income_analysis": "J'ai détecté une demande d'analyse de revenus. Je vais analyser vos revenus.",
        "unclear_intent": "Je n'ai pas bien compris votre demande. Pouvez-vous être plus spécifique ?"
    }
    
    base_response = intent_responses.get(intent_result.intent.value, "Intention détectée")
    
    # Ajout des entités détectées
    entities_info = []
    if intent_result.entities.merchant:
        entities_info.append(f"Marchand: {intent_result.entities.merchant}")
    if intent_result.entities.category:
        entities_info.append(f"Catégorie: {intent_result.entities.category}")
    if intent_result.entities.amount:
        entities_info.append(f"Montant: {intent_result.entities.amount}")
    if intent_result.entities.period:
        entities_info.append(f"Période: {intent_result.entities.period}")
    
    if entities_info:
        base_response += f" ({', '.join(entities_info)})"
    
    return base_response

def _get_clarification_message(intent_result) -> str:
    """Génère un message de clarification pour les intentions peu claires"""
    
    if intent_result.intent.value == "unclear_intent":
        return "Pouvez-vous préciser ce que vous recherchez ? Par exemple : 'mes achats Netflix', 'mes restaurants ce mois', ou 'plus de 100€'."
    
    return f"Je ne suis pas sûr de votre demande (confiance: {intent_result.confidence:.0%}). Pouvez-vous être plus précis ?"

async def _log_conversation_analytics(request: ChatRequest, response: ChatResponse):
    """Tâche en arrière-plan pour logger les analytics"""
    try:
        # TODO: Implémenter analytics si nécessaire
        logger.debug(f"Analytics - User: {request.user_id}, Intent: {response.intent}, Confidence: {response.confidence}")
    except Exception as e:
        logger.error(f"Erreur analytics: {str(e)}")