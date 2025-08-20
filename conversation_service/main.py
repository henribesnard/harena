"""FastAPI application entrypoint for the conversation service.

This module creates the FastAPI application, applies common middleware and
includes the API routers.  It exposes an ``app`` object that can be used by
ASGI servers such as Uvicorn.
"""

from fastapi import FastAPI

from .api.routes import router as api_router
from .api.websocket import router as websocket_router
from .api.middleware import setup_middleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Conversation Service")
    setup_middleware(app)
    app.include_router(api_router)
    app.include_router(websocket_router)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("conversation_service.main:app", host="0.0.0.0", port=8000, reload=True)
