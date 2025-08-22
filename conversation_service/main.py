"""Application entry point for the conversation service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from conversation_service.api.middleware import setup_middleware
from conversation_service.api.routes import router as api_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(title="Conversation Service")

    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_middleware(app)
    app.include_router(api_router)

    return app


app = create_app()
