"""
Routes API pour conversation service - Architecture v2.0 complete
Endpoints v1 avec backend v2.0
"""
import logging
import time
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, AgentMetrics, ConversationResponseError
)

# Architecture v2.0 imports
from conversation_service.api.dependencies import (
    get_config_manager,
    get_conversation_orchestrator,
    get_application_health,
    get_pipeline_stats,
    app_state,
    # Legacy compatibility
    get_deepseek_client,
    get_cache_manager,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency,
    get_conversation_persistence
)
from conversation_service.api.middleware.auth_middleware import get_current_user_id, get_current_jwt_token
from conversation_service.services.conversation_persistence import (
    ConversationPersistenceService, create_conversation_data, create_turn_data
)
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.validation_utils import validate_user_message, sanitize_user_input

# Configuration du router et logger
router = APIRouter(
    prefix="/api/v1/conversation",
    tags=["conversation"],
    responses={
        500: {"description": "Erreur interne"},
        503: {"description": "Service indisponible"}
    }
)
logger = logging.getLogger(__name__)

@router.post("/{path_user_id}")
async def analyze_conversation(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence),
    _rate_limit: None = Depends(rate_limit_dependency),
    jwt_token: str = Depends(get_current_jwt_token)
):
    """
    Endpoint principal conversation service v2.0
    
    Pipeline 5 etapes:
    1. Context Management
    2. Intent Classification
    3. Query Building
    4. Query Execution
    5. Response Generation
    """
    
    start_time = time.time()
    request_id = f"conv_{int(time.time() * 1000)}_{validated_user_id}"
    
    try:
        # Validation du message
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        clean_message = sanitize_user_input(request_data.message)
        
        # Utiliser le nouvel orchestrateur v2.0 si disponible
        if app_state.initialized and app_state.conversation_orchestrator:
            from conversation_service.core.conversation_orchestrator import ConversationRequest as OrchestratorRequest
            
            orchestrator_request = OrchestratorRequest(
                user_id=validated_user_id,
                user_message=clean_message,
                conversation_id=getattr(request_data, 'conversation_id', None),
                jwt_token=jwt_token
            )
            
            result = await app_state.conversation_orchestrator.process_conversation(orchestrator_request)
            
            # Conversion vers format API legacy
            response = _convert_orchestrator_to_api_response(
                result,
                validated_user_id,
                clean_message,
                request_id,
                int((time.time() - start_time) * 1000)
            )
            
            # Persistence si disponible
            if persistence_service:
                await _save_conversation_turn(
                    persistence_service=persistence_service,
                    user_id=validated_user_id,
                    user_message=clean_message,
                    assistant_response=result.response_text if result.response_text else "Erreur de traitement",
                    result_data=_serialize_conversation_result(result),
                    request_id=request_id
                )
            
            return response
        
        else:
            # Fallback vers l'ancien systeme legacy
            logger.warning(f"[{request_id}] v2.0 orchestrator not available, using legacy system")
            return await _process_legacy_conversation(
                user_id=validated_user_id,
                message=clean_message,
                request=request,
                user_context=user_context,
                persistence_service=persistence_service,
                request_id=request_id,
                start_time=start_time
            )
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{request_id}] Erreur traitement conversation: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur traitement conversation",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.post("/{path_user_id}/stream")
async def analyze_conversation_stream(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence),
    _rate_limit: None = Depends(rate_limit_dependency),
    jwt_token: str = Depends(get_current_jwt_token)
):
    """
    Endpoint streaming v2.0 - Stream la reponse finale
    """

    async def generate_stream() -> AsyncGenerator[str, None]:
        request_id = f"stream_{int(time.time() * 1000)}_{validated_user_id}"
        accumulated_response = ""

        try:
            # Validation
            message_validation = validate_user_message(request_data.message)
            if not message_validation["valid"]:
                error_data = {
                    "error": "Message invalide",
                    "details": message_validation["errors"]
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            clean_message = sanitize_user_input(request_data.message)

            # Utiliser orchestrateur v2.0 si disponible
            if app_state.initialized and app_state.conversation_orchestrator:
                from conversation_service.core.conversation_orchestrator import ConversationRequest as OrchestratorRequest

                orchestrator_request = OrchestratorRequest(
                    user_id=validated_user_id,
                    user_message=clean_message,
                    conversation_id=getattr(request_data, 'conversation_id', None),
                    jwt_token=jwt_token
                )

                # Stream de la reponse et accumuler le contenu
                async for chunk in app_state.conversation_orchestrator.process_conversation_stream(orchestrator_request):
                    # Accumuler les chunks de réponse
                    if chunk.get("type") == "response_chunk":
                        accumulated_response += chunk.get("content", "")

                    # chunk est maintenant un dictionnaire, on le serialise en SSE
                    yield f"data: {json.dumps(chunk)}\n\n"

                    # Si c'est une erreur ou la fin, on arrête
                    if chunk.get("type") in ["error", "response_end"]:
                        break

                # Sauvegarder la conversation après le streaming
                if persistence_service and accumulated_response:
                    try:
                        await _save_conversation_turn(
                            persistence_service=persistence_service,
                            user_id=validated_user_id,
                            user_message=clean_message,
                            assistant_response=accumulated_response,
                            result_data={"request_id": request_id, "streaming": True},
                            request_id=request_id
                        )
                        logger.info(f"[{request_id}] Conversation sauvegardée en base")
                    except Exception as e:
                        logger.error(f"[{request_id}] Erreur sauvegarde conversation: {str(e)}")

            else:
                # Fallback: pas de streaming pour legacy
                yield f"data: {json.dumps({'error': 'Streaming non disponible en mode legacy'})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Erreur streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@router.websocket("/{user_id}/ws")
async def websocket_conversation(websocket: WebSocket, user_id: int):
    """
    WebSocket endpoint pour conversations temps reel v2.0
    """
    await websocket.accept()
    
    try:
        if not (app_state.initialized and app_state.conversation_orchestrator):
            await websocket.send_json({
                "type": "error",
                "message": "WebSocket non disponible - v2.0 orchestrator non initialise"
            })
            await websocket.close()
            return
        
        while True:
            # Recevoir message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message = data.get("message", "")
                
                # Validation
                message_validation = validate_user_message(message)
                if not message_validation["valid"]:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Message invalide",
                        "errors": message_validation["errors"]
                    })
                    continue
                
                clean_message = sanitize_user_input(message)
                
                # Traitement via orchestrateur
                from conversation_service.core.conversation_orchestrator import ConversationRequest as OrchestratorRequest
                
                orchestrator_request = OrchestratorRequest(
                    user_id=user_id,
                    user_message=clean_message,
                    conversation_id=data.get("conversation_id")
                )
                
                # Stream via WebSocket
                async for chunk in app_state.conversation_orchestrator.process_conversation_stream(orchestrator_request):
                    await websocket.send_json(chunk)
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Erreur WebSocket: {str(e)}"
            })
        except:
            pass
        await websocket.close()

@router.get("/health")
async def conversation_health_detailed():
    """Health check conversation service v2.0"""
    try:
        if app_state.initialized and app_state.conversation_orchestrator:
            health = await get_application_health()
            
            # Corriger le status global basé sur les composants individuels
            components = health.get("components", {})
            all_healthy = all(
                comp.get("status") == "healthy" 
                for comp in components.values()
            )
            
            # Si tous les composants sont healthy, le service est healthy
            global_status = "healthy" if all_healthy else health.get("status", "unhealthy")
            
            return {
                "service": "conversation_service",
                "version": "2.0",
                "architecture": "complete",
                "status": global_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": components,
                "features": {
                    "intent_classification": True,
                    "entity_extraction": True,
                    "context_management": True,
                    "query_building": True,
                    "search_execution": True,
                    "response_generation": True,
                    "streaming": True,
                    "websocket": True,
                    "monitoring": True
                }
            }
        else:
            # Fallback health check legacy
            health_metrics = metrics_collector.get_health_metrics()
            return {
                "service": "conversation_service",
                "version": "1.0",
                "architecture": "legacy",
                "status": health_metrics.get("status", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "warning": "v2.0 architecture not initialized"
            }
    
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return {
            "service": "conversation_service",
            "version": "unknown",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/status")
async def conversation_status():
    """Status global service v2.0"""
    try:
        if app_state.initialized and app_state.conversation_orchestrator:
            stats = await get_pipeline_stats()
            health = await get_application_health()
            
            return {
                "status": health.get("status", "healthy"),
                "version": "2.0",
                "architecture": "complete",
                "ready": health.get("status") == "healthy",
                "pipeline_stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            health_metrics = metrics_collector.get_health_metrics()
            return {
                "status": health_metrics.get("status", "unknown"),
                "version": "1.0",
                "architecture": "legacy",
                "ready": health_metrics.get("status") == "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    except Exception as e:
        logger.error(f"Erreur status check: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/metrics")
async def conversation_metrics():
    """Métriques détaillées v2.0"""
    try:
        if app_state.initialized and app_state.conversation_orchestrator:
            stats = await get_pipeline_stats()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "v2.0",
                "pipeline_metrics": stats,
                "service_info": {
                    "name": "conversation_service",
                    "version": "2.0",
                    "architecture": "complete"
                }
            }
        else:
            # Fallback legacy metrics
            all_metrics = metrics_collector.get_all_metrics()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "legacy",
                "metrics": all_metrics,
                "service_info": {
                    "name": "conversation_service",
                    "version": "1.0",
                    "architecture": "legacy"
                }
            }
    
    except Exception as e:
        logger.error(f"Erreur export métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur récupération métriques",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/conversations/{user_id}")
async def get_user_conversations(
    user_id: int,
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence),
    limit: int = Query(50, description="Nombre maximum de conversations")
):
    """Récupération des conversations d'un utilisateur"""
    try:
        if validated_user_id != user_id:
            raise HTTPException(status_code=403, detail="Accès non autorisé")

        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service persistence non disponible")

        conversations = persistence_service.get_user_conversations(user_id, limit=limit)

        return {
            "user_id": user_id,
            "conversations": [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "turn_count": conv.total_turns
                }
                for conv in conversations
            ],
            "total_count": len(conversations)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")

@router.get("/conversation/{conversation_id}/turns")
async def get_conversation_turns(
    conversation_id: int,
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence)
):
    """Récupération d'une conversation complète avec tous ses tours"""
    try:
        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service persistence non disponible")

        conversation = persistence_service.get_conversation_with_turns(
            conversation_id=conversation_id,
            user_id=validated_user_id
        )

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation non trouvée")

        return {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "total_turns": conversation.total_turns
            },
            "turns": [
                {
                    "id": turn.id,
                    "turn_number": turn.turn_number,
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "created_at": turn.created_at.isoformat()
                }
                for turn in conversation.turns
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")

@router.get("/admin/conversations/count")
async def count_all_conversations(
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence)
):
    """ADMIN: Compte le nombre total de conversations"""
    try:
        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service persistence non disponible")

        from db_service.models.conversation import ConversationTurn, Conversation

        total_conversations = persistence_service.db.query(Conversation).count()
        total_turns = persistence_service.db.query(ConversationTurn).count()

        # Count by user
        from sqlalchemy import func
        by_user = persistence_service.db.query(
            Conversation.user_id,
            func.count(Conversation.id).label('count')
        ).group_by(Conversation.user_id).all()

        return {
            "total_conversations": total_conversations,
            "total_turns": total_turns,
            "by_user": [{"user_id": u, "count": c} for u, c in by_user]
        }

    except Exception as e:
        logger.error(f"Erreur comptage conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")

@router.delete("/admin/conversations/clear")
async def clear_all_conversations_admin(
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence)
):
    """ADMIN: Supprime toutes les conversations (pour dev/test uniquement)"""
    try:
        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service persistence non disponible")

        # Delete all turns first (FK constraint)
        from db_service.models.conversation import ConversationTurn, Conversation
        turns_deleted = persistence_service.db.query(ConversationTurn).delete()

        # Then delete all conversations
        conv_deleted = persistence_service.db.query(Conversation).delete()

        persistence_service.db.commit()

        logger.info(f"Admin cleared all conversations: {conv_deleted} conversations, {turns_deleted} turns")

        return {
            "success": True,
            "conversations_deleted": conv_deleted,
            "turns_deleted": turns_deleted,
            "message": "Toutes les conversations ont été supprimées"
        }

    except Exception as e:
        persistence_service.db.rollback()
        logger.error(f"Erreur suppression conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")

# === UTILITAIRES ===

def _serialize_conversation_result(result) -> Dict[str, Any]:
    """Sérialise un ConversationResult pour la persistence en évitant les erreurs JSON"""
    import json
    from datetime import datetime
    from enum import Enum
    
    def convert_value(obj):
        """Convertit les objets non-sérialisables en JSON"""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return {k: convert_value(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_value(item) for item in obj]
        else:
            return obj
    
    try:
        # Conversion récursive de tous les attributs
        serialized = convert_value(result.__dict__)
        
        # Test de sérialisation pour vérifier que tout est compatible JSON
        json.dumps(serialized)
        
        return serialized
    except Exception as e:
        logger.warning(f"Erreur sérialisation ConversationResult: {str(e)}")
        # Fallback avec données minimales
        return {
            "success": getattr(result, 'success', False),
            "response_text": getattr(result, 'response_text', ''),
            "conversation_id": getattr(result, 'conversation_id', ''),
            "pipeline_stage": getattr(result, 'pipeline_stage', '').value if hasattr(getattr(result, 'pipeline_stage', ''), 'value') else 'unknown',
            "error_message": getattr(result, 'error_message', None),
            "serialization_error": str(e)
        }

def _convert_orchestrator_to_api_response(
    result,
    user_id: int,
    message: str,
    request_id: str,
    processing_time_ms: int
) -> Dict[str, Any]:
    """Convertit la reponse de l'orchestrateur vers le format API legacy"""
    
    return {
        "user_id": user_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "processing_time_ms": processing_time_ms,
        "status": "completed" if result.success else "error",

        "intent": {
            "type": result.classified_intent.get("intent_group") if result.classified_intent else "UNKNOWN",
            "confidence": result.classified_intent.get("confidence", 0.0) if result.classified_intent else 0.0,
            "supported": True,  # Si on arrive ici, c'est que c'est supporté
            "entities": result.classified_intent.get("entities", []) if result.classified_intent else []
        } if result.classified_intent else None,

        "query": result.built_query if hasattr(result, 'built_query') and result.built_query else None,

        "response": {
            "message": result.response_text,
            "structured_data": result.insights
        },

        "search_summary": {
            "found_results": bool(result.search_results and len(result.search_results) > 0),
            "total_results": result.search_total_hits
        },

        "performance": {
            "overall_success": result.success,
            "pipeline_stages": len(result.pipeline_metrics) if hasattr(result, 'pipeline_metrics') else 5
        },

        "architecture": "v2.0"
    }

async def _process_legacy_conversation(
    user_id: int,
    message: str,
    request: Request,
    user_context: Dict[str, Any],
    persistence_service: Optional[ConversationPersistenceService],
    request_id: str,
    start_time: float
) -> Dict[str, Any]:
    """Traitement conversation legacy (fallback)"""
    
    # Import legacy components si necessaire
    try:
        from conversation_service.clients.deepseek_client import DeepSeekClient
        deepseek_client = request.app.state.deepseek_client
        
        # Traitement basique legacy
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "processing_time_ms": processing_time_ms,
            "status": "completed",
            "response": {
                "message": "Service en mode legacy - fonctionnalites limitees",
                "structured_data": None
            },
            "architecture": "legacy"
        }
    
    except Exception as e:
        logger.error(f"[{request_id}] Erreur traitement legacy: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporairement indisponible",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

async def _save_conversation_turn(
    persistence_service: ConversationPersistenceService,
    user_id: int,
    user_message: str,
    assistant_response: str,
    result_data: Dict[str, Any],
    request_id: str
):
    """Sauvegarde un tour de conversation"""
    try:
        conversation = persistence_service.get_or_create_active_conversation(
            user_id=user_id,
            conversation_title=f"Conversation du {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        
        turn_data = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": "v2.0",
            "pipeline_data": result_data
        }
        
        persistence_service.add_conversation_turn(
            conversation_id=conversation.id,
            user_message=user_message,
            assistant_response=assistant_response,
            turn_data=turn_data
        )
        
        logger.debug(f"[{request_id}] Conversation sauvegardee - ID: {conversation.id}")
    
    except Exception as e:
        logger.error(f"[{request_id}] Erreur sauvegarde conversation: {str(e)}")

logger.info("Conversation routes v2.0 configured")