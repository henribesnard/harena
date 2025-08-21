"""Application entrypoint for the Harena FastAPI service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.websocket import router as ws_router
from config.settings import settings


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

    app.include_router(api_router)
    app.include_router(ws_router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
