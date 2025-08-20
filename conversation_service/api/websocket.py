import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from .dependencies import get_team_manager, get_conversation_manager
from ..core.conversation_manager import ConversationManager
from ..core.mvp_team_manager import MVPTeamManager

logger = logging.getLogger(__name__)

ws_router = APIRouter()


async def agent_stream(
    team_manager: MVPTeamManager,
    conversation_manager: ConversationManager,
    user_id: int,
    conversation_id: str,
    user_message: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run agents sequentially and yield incremental results."""
    start = time.perf_counter()

    intent_agent = team_manager.agents.get("intent_agent")
    search_agent = team_manager.agents.get("search_query_agent")
    response_agent = team_manager.agents.get("response_agent")

    await conversation_manager.update_user_context(conversation_id, user_id, user_message)

    intent_response = await intent_agent.execute_with_metrics(
        {"user_message": user_message}, user_id
    )
    intent_result = intent_response.metadata.get("intent_result") if intent_response.metadata else None
    yield {"event": "intent", "data": intent_response.metadata}

    search_response = None
    if intent_result and getattr(intent_result, "search_required", True):
        search_response = await search_agent.execute_with_metrics(
            {"intent_result": intent_result, "user_message": user_message}, user_id
        )
        yield {"event": "search", "data": search_response.metadata}

    context = await conversation_manager.get_context(conversation_id)
    response_payload = {
        "user_message": user_message,
        "search_results": search_response,
        "context": context,
        "search_error": False if search_response is None else not search_response.success,
    }
    response_response = await response_agent.execute_with_metrics(response_payload, user_id)
    yield {
        "event": "response",
        "message": response_response.content,
        "metadata": response_response.metadata,
    }

    total_time = (time.perf_counter() - start) * 1000
    try:
        await conversation_manager.add_turn(
            conversation_id,
            user_id,
            user_message,
            response_response.content,
            intent_result=intent_result,
            processing_time_ms=total_time,
            agent_chain=["intent_agent", "search_query_agent", "response_agent"],
            search_results_count=(
                search_response.metadata.get("search_results_count")
                if search_response and search_response.metadata
                else None
            ),
            confidence_score=response_response.confidence_score,
        )
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")


@ws_router.websocket("/ws/chat")
async def chat_websocket(
    websocket: WebSocket,
    team_manager: Annotated[MVPTeamManager, Depends(get_team_manager)],
    conversation_manager: Annotated[ConversationManager, Depends(get_conversation_manager)],
):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            message: str = data.get("message", "")
            conversation_id: str = data.get("conversation_id", "default")
            user_id: int = int(data.get("user_id", 0))

            async for event in agent_stream(
                team_manager, conversation_manager, user_id, conversation_id, message
            ):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"event": "error", "message": str(e)})
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")
"""WebSocket endpoints for the conversation service."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Simple echo websocket used in tests."""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(message)
    except WebSocketDisconnect:
        pass
