"""
Conversation endpoints for v3 API - Compatible with v1 format
WITH PERSISTENCE - Saves conversations to PostgreSQL
"""
import logging
import time
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime, timezone

from ....core.agent_orchestrator import AgentOrchestrator
from ....models import UserQuery, ConversationResponse
from ....config.settings import settings
from ....services.conversation_persistence import (
    ConversationPersistenceService,
    create_turn_metadata_v3
)
from ....api.dependencies import get_persistence_service, extract_jwt_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/conversation", tags=["conversation"])

# Instance globale de l'orchestrateur
orchestrator: Optional[AgentOrchestrator] = None


class ClientInfo(BaseModel):
    """Client information"""
    platform: str = "web"
    version: str = "1.0.0"


class ConversationRequest(BaseModel):
    """Requête de conversation - Compatible v1 format"""
    client_info: Optional[ClientInfo] = None
    message: str
    message_type: str = "text"
    priority: str = "normal"
    conversation_id: Optional[int] = None


class ConversationResponseModel(BaseModel):
    """Modèle de réponse de conversation"""
    success: bool
    message: str
    total_results: Optional[int] = None
    aggregations_summary: Optional[str] = None
    metadata: dict = {}


def get_orchestrator() -> AgentOrchestrator:
    """Dépendance pour obtenir l'orchestrateur"""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator(
            search_service_url=settings.SEARCH_SERVICE_URL,
            max_correction_attempts=settings.MAX_CORRECTION_ATTEMPTS,
            llm_model=settings.LLM_MODEL
        )
    return orchestrator


@router.post("/{user_id}", response_model=Dict[str, Any])
async def analyze_conversation(
    user_id: int,
    request: ConversationRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
    persistence: ConversationPersistenceService = Depends(get_persistence_service),
    jwt_token: Optional[str] = Depends(extract_jwt_token)
) -> Dict[str, Any]:
    """
    Endpoint principal compatible v1 avec PERSISTENCE - POST /api/v3/conversation/{user_id}

    Le pipeline complet avec agents autonomes:
    1. Analyse la question (QueryAnalyzerAgent)
    2. Construit la query Elasticsearch (ElasticsearchBuilderAgent)
    3. Exécute la query sur search_service
    4. Auto-correction si échec
    5. Génère la réponse (ResponseGeneratorAgent)
    6. Sauvegarde la conversation en base de données ✨ NEW

    Args:
        user_id: ID de l'utilisateur (path parameter)
        request: Corps de la requête avec message, client_info, etc.
        orch: Orchestrateur d'agents (injecté)
        persistence: Service de persistence (injecté)
        jwt_token: Token JWT extrait de l'en-tête Authorization

    Returns:
        Réponse conversationnelle avec résultats et insights
    """
    start_time = time.time()

    try:
        logger.info(f"Received conversation request from user {user_id}")

        # Créer la requête utilisateur
        user_query = UserQuery(
            user_id=user_id,
            message=request.message,
            conversation_id=str(request.conversation_id) if request.conversation_id else None,
            context=[]
        )

        # Traiter la requête via l'orchestrateur
        response: ConversationResponse = await orch.process_query(
            user_query=user_query,
            jwt_token=jwt_token
        )

        # === PERSISTENCE: Sauvegarder la conversation ===
        saved_conversation_id = None
        try:
            # Récupérer ou créer la conversation
            conversation = persistence.get_or_create_conversation(
                user_id=user_id,
                conversation_id=request.conversation_id
            )

            # Créer les métadonnées du tour
            processing_time_ms = int((time.time() - start_time) * 1000)
            turn_metadata = create_turn_metadata_v3(
                user_query=request.message,
                query_analysis=response.metadata.get("query_analysis"),
                elasticsearch_query=response.metadata.get("elasticsearch_query"),
                search_results_summary={
                    "total": response.search_results.total if response.search_results else 0,
                    "aggregations_summary": response.aggregations_summary
                } if response.search_results else None,
                processing_time_ms=processing_time_ms,
                corrections_applied=response.metadata.get("corrections_applied", 0)
            )

            # Ajouter le tour à la conversation
            persistence.add_conversation_turn(
                conversation_id=conversation.id,
                user_message=request.message,
                assistant_response=response.message,
                turn_data=turn_metadata
            )

            saved_conversation_id = conversation.id
            logger.info(f"✅ Conversation saved - ID: {saved_conversation_id}, User: {user_id}")

        except Exception as persist_error:
            # Ne pas faire échouer la requête si la persistence échoue
            logger.error(f"❌ Failed to save conversation: {persist_error}", exc_info=True)

        # Format de réponse compatible v1
        api_response = {
            "user_id": user_id,
            "message": request.message,
            "status": "completed" if response.success else "error",
            "response": {
                "message": response.message,
                "structured_data": {
                    "total_results": response.search_results.total if response.search_results else 0,
                    "aggregations_summary": response.aggregations_summary
                }
            },
            "search_summary": {
                "found_results": response.search_results is not None and response.search_results.total > 0,
                "total_results": response.search_results.total if response.search_results else 0
            },
            "metadata": response.metadata,
            "architecture": "v3_langchain_agents"
        }

        # Ajouter l'ID de conversation si sauvegardé
        if saved_conversation_id:
            api_response["conversation_id"] = saved_conversation_id

        return api_response

    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/{user_id}/stream")
async def analyze_conversation_stream(
    user_id: int,
    request: ConversationRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator),
    persistence: ConversationPersistenceService = Depends(get_persistence_service),
    jwt_token: Optional[str] = Depends(extract_jwt_token)
):
    """
    Endpoint streaming RÉEL avec PERSISTENCE - POST /api/v3/conversation/{user_id}/stream

    Stream la réponse token par token comme v1
    """
    from fastapi.responses import StreamingResponse
    import json

    async def generate_stream():
        start_time = time.time()
        saved_conversation_id = None
        accumulated_response = ""

        try:
            # Créer la requête utilisateur
            user_query = UserQuery(
                user_id=user_id,
                message=request.message,
                conversation_id=str(request.conversation_id) if request.conversation_id else None,
                context=[]
            )

            # === ÉTAPE 0: Routage d'intention (NOUVEAU) ===
            # 🔹 Message de progression UX
            yield f"data: {json.dumps({'type': 'status', 'message': '• Analyse de votre question...'})}\n\n"

            logger.info("Step 0: Intent classification (stream)")
            intent_response = await orch.intent_router.classify_intent(user_query)

            if not intent_response.success:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to classify intent'})}\n\n"
                return

            intent_classification = intent_response.data
            logger.info(f"Intent classified (stream): {intent_classification.category.value}, requires_search={intent_classification.requires_search}")

            # === CAS 1: Réponse conversationnelle (pas de recherche) ===
            if not intent_classification.requires_search:
                logger.info("Conversational intent detected (stream), responding directly")

                # 🔹 Message de progression UX
                yield f"data: {json.dumps({'type': 'status', 'message': '• Préparation de la réponse...'})}\n\n"

                # Utiliser la réponse suggérée ou générer une réponse persona
                if intent_classification.suggested_response:
                    response_text = intent_classification.suggested_response
                else:
                    response_text = orch.intent_router.get_persona_response(
                        intent_classification.category
                    )

                # Envoyer response_start
                yield f"data: {json.dumps({'type': 'response_start'})}\n\n"

                # Streamer la réponse mot par mot pour simuler le streaming
                words = response_text.split(' ')
                for i, word in enumerate(words):
                    chunk = word + (' ' if i < len(words) - 1 else '')
                    accumulated_response += chunk
                    chunk_data = {'type': 'response_chunk', 'content': chunk}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.05)  # Petit délai pour effet de streaming

                processing_time_ms = int((time.time() - start_time) * 1000)

                # Envoyer response_end
                end_metadata = {
                    'type': 'response_end',
                    'metadata': {
                        'total_results': 0,
                        'response_length': len(accumulated_response),
                        'processing_time_ms': processing_time_ms,
                        'intent': intent_classification.category.value,
                        'requires_search': False
                    }
                }
                yield f"data: {json.dumps(end_metadata)}\n\n"
                return

            # === CAS 2: Pipeline financier complet (recherche requise) ===
            logger.info("Financial intent detected (stream), proceeding with search pipeline")

            # 🔹 Message de progression UX
            yield f"data: {json.dumps({'type': 'status', 'message': '• Recherche de vos transactions...'})}\n\n"

            # === ÉTAPE 1-3: Pipeline jusqu'à la récupération des résultats ===
            logger.info("Step 1: Analyzing user query")
            analysis_response = await orch.query_analyzer.analyze(user_query)

            if not analysis_response.success:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to analyze query'})}\n\n"
                return

            query_analysis = analysis_response.data

            logger.info("Step 2: Building Elasticsearch query")
            build_response = await orch.query_builder.build_query(
                query_analysis, user_query
            )

            if not build_response.success:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to build query'})}\n\n"
                return

            es_query = build_response.data

            # 🔹 Message de progression UX
            yield f"data: {json.dumps({'type': 'status', 'message': '• Analyse de vos données...'})}\n\n"

            logger.info("Step 3: Executing query on search_service")
            search_results = await orch._execute_query(es_query, user_query.user_id, jwt_token)

            if not search_results:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to execute query'})}\n\n"
                return

            # 🔹 Message de progression UX basé sur les résultats
            if search_results.total > 0:
                yield f"data: {json.dumps({'type': 'status', 'message': f'• {search_results.total} transaction(s) trouvée(s), génération de la réponse...'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': '• Préparation de la réponse...'})}\n\n"

            # Envoyer response_start
            yield f"data: {json.dumps({'type': 'response_start'})}\n\n"

            # === PERSISTENCE PRÉCOCE: Créer ou récupérer la conversation ===
            try:
                conversation = persistence.get_or_create_conversation(
                    user_id=user_id,
                    conversation_id=request.conversation_id
                )
                saved_conversation_id = conversation.id

                # Envoyer l'ID de conversation immédiatement
                conv_id_data = {'type': 'conversation_id', 'conversation_id': saved_conversation_id}
                yield f"data: {json.dumps(conv_id_data)}\n\n"
            except Exception as persist_error:
                logger.error(f"❌ Failed to create conversation: {persist_error}", exc_info=True)

            # === ÉTAPE 4: Stream de la réponse ===
            async for chunk in orch.response_generator.generate_response_stream(
                user_message=user_query.message,
                search_results=search_results,
                original_query_analysis=query_analysis.__dict__ if hasattr(query_analysis, '__dict__') else None
            ):
                accumulated_response += chunk
                chunk_data = {'type': 'response_chunk', 'content': chunk}
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Calculer le temps de traitement
            processing_time_ms = int((time.time() - start_time) * 1000)

            # === PERSISTENCE FINALE: Sauvegarder le tour complet ===
            if saved_conversation_id and accumulated_response:
                try:
                    turn_metadata = create_turn_metadata_v3(
                        user_query=request.message,
                        query_analysis=query_analysis.__dict__ if hasattr(query_analysis, '__dict__') else None,
                        elasticsearch_query=es_query.__dict__ if hasattr(es_query, '__dict__') else None,
                        search_results_summary={
                            "total": search_results.total,
                            "aggregations_summary": orch.response_generator._format_aggregations(search_results.aggregations) if search_results.aggregations else None
                        },
                        processing_time_ms=processing_time_ms,
                        corrections_applied=0
                    )

                    persistence.add_conversation_turn(
                        conversation_id=saved_conversation_id,
                        user_message=request.message,
                        assistant_response=accumulated_response,
                        turn_data=turn_metadata
                    )
                    logger.info(f"✅ Conversation saved (stream) - ID: {saved_conversation_id}, User: {user_id}")

                except Exception as persist_error:
                    logger.error(f"❌ Failed to save turn: {persist_error}", exc_info=True)

            # Envoyer response_end avec metadata
            end_data = {
                'type': 'response_end',
                'metadata': {
                    'total_results': search_results.total,
                    'response_length': len(accumulated_response),
                    'processing_time_ms': processing_time_ms
                }
            }
            yield f"data: {json.dumps(end_data)}\n\n"

        except Exception as e:
            logger.error(f"Error in stream: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/health")
async def conversation_health():
    """
    Health check compatible v1 - GET /api/v1/conversation/health
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            health = await orchestrator.health_check()
            return {
                "service": "conversation_service_v3",
                "version": "3.0.0",
                "architecture": "langchain_agents",
                "status": "healthy" if health.get("healthy", False) else "degraded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": health
            }
        else:
            return {
                "service": "conversation_service_v3",
                "version": "3.0.0",
                "status": "initializing",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "service": "conversation_service_v3",
            "version": "3.0.0",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/status")
async def conversation_status():
    """
    Status endpoint compatible v1 - GET /api/v1/conversation/status
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            stats = orchestrator.get_stats()
            return {
                "status": "healthy",
                "version": "3.0.0",
                "architecture": "langchain_agents",
                "ready": True,
                "stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "initializing",
                "ready": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/metrics")
async def conversation_metrics():
    """
    Metrics endpoint compatible v1 - GET /api/v1/conversation/metrics
    """
    from datetime import datetime, timezone

    try:
        if orchestrator:
            stats = orchestrator.get_stats()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "v3_langchain_agents",
                "metrics": stats,
                "service_info": {
                    "name": "conversation_service_v3",
                    "version": "3.0.0",
                    "features": [
                        "autonomous_agents",
                        "auto_correction",
                        "elasticsearch_schema_aware",
                        "aggregations_support"
                    ]
                }
            }
        else:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Orchestrator not initialized"
            }
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PERSISTENCE ENDPOINTS - History and Conversation Retrieval
# ============================================================================

@router.get("/conversations/{user_id}")
async def get_user_conversations(
    user_id: int,
    persistence: ConversationPersistenceService = Depends(get_persistence_service),
    jwt_token: Optional[str] = Depends(extract_jwt_token),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
) -> Dict[str, Any]:
    """
    Récupère l'historique des conversations d'un utilisateur
    Compatible v1 - GET /api/v3/conversation/conversations/{user_id}

    Args:
        user_id: ID de l'utilisateur
        persistence: Service de persistence (injecté)
        jwt_token: Token JWT (pour authentification future)
        limit: Nombre maximum de conversations à récupérer
        offset: Décalage pour la pagination

    Returns:
        Liste des conversations avec métadonnées
    """
    try:
        logger.info(f"Fetching conversations for user {user_id} (limit: {limit}, offset: {offset})")

        # Récupérer les conversations
        conversations = persistence.get_user_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        # Formater la réponse
        conversations_data = []
        for conv in conversations:
            conversations_data.append({
                "id": conv.id,
                "title": conv.title,
                "total_turns": conv.total_turns,
                "status": conv.status,
                "last_activity_at": conv.last_activity_at.isoformat() if conv.last_activity_at else None,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "data": conv.data
            })

        return {
            "user_id": user_id,
            "conversations": conversations_data,
            "total": len(conversations_data),
            "limit": limit,
            "offset": offset,
            "architecture": "v3_langchain_agents"
        }

    except Exception as e:
        logger.error(f"Error fetching conversations for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")


@router.get("/conversation/{conversation_id}/turns")
async def get_conversation_turns(
    conversation_id: int,
    persistence: ConversationPersistenceService = Depends(get_persistence_service),
    jwt_token: Optional[str] = Depends(extract_jwt_token)
) -> Dict[str, Any]:
    """
    Récupère les détails d'une conversation avec tous ses tours
    Compatible v1 - GET /api/v3/conversation/conversation/{conversation_id}/turns

    Args:
        conversation_id: ID de la conversation
        persistence: Service de persistence (injecté)
        jwt_token: Token JWT (pour authentification future)

    Returns:
        Conversation avec tous les tours (messages)
    """
    try:
        logger.info(f"Fetching conversation {conversation_id} with turns")

        # Récupérer la conversation avec ses tours
        # Note: Pour la sécurité, on devrait vérifier que l'utilisateur a accès à cette conversation
        # via le JWT token. Pour l'instant, on fait confiance au frontend.
        conversation = persistence.get_conversation_with_turns(
            conversation_id=conversation_id,
            user_id=None  # TODO: Extract user_id from JWT token for security
        )

        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

        # Formater les tours
        turns_data = []
        for turn in conversation.turns:
            turns_data.append({
                "id": turn.id,
                "turn_number": turn.turn_number,
                "user_message": turn.user_message,
                "assistant_response": turn.assistant_response,
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
                "data": turn.data
            })

        return {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "total_turns": conversation.total_turns,
                "status": conversation.status,
                "last_activity_at": conversation.last_activity_at.isoformat() if conversation.last_activity_at else None,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "user_id": conversation.user_id
            },
            "turns": turns_data,
            "architecture": "v3_langchain_agents"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation: {str(e)}")
