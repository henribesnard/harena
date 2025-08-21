"""Router exposing realtime conversation endpoints."""

from typing import List

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from .dependencies import get_session_id
from ..agents.response_generator import stream_response

router = APIRouter()

# Connected websocket clients
connections: List[WebSocket] = []


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Depends(get_session_id),
) -> None:
    """Basic websocket endpoint broadcasting generated responses.

    Each message received is passed to the :func:`stream_response` generator and
    every chunk produced is broadcast to all connected clients.
    """
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            async for chunk in stream_response(message):
                # broadcast
                for ws in list(connections):
                    try:
                        await ws.send_text(chunk)
                    except WebSocketDisconnect:
                        connections.remove(ws)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connections:
            connections.remove(websocket)
