from fastapi import WebSocket, HTTPException, status

async def get_session_id(websocket: WebSocket) -> str:
    """Authenticate websocket connections using a session token.

    The token is expected in the query string as ``?session=<token>``. If the
    token is missing the connection is closed with an appropriate code and a
    403 error is raised so the dependency chain is halted.
    """
    session_id = websocket.query_params.get("session")
    if not session_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session non authentifiée"
        )
    return session_id
