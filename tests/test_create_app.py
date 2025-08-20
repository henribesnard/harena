import importlib
import sys
import types
import builtins
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware


def build_app() -> FastAPI:
    routes = types.ModuleType("routes")
    routes.router = APIRouter()

    @routes.router.get("/chat")
    async def chat_endpoint():
        return {"ok": True}

    routes.websocket_router = APIRouter()

    @routes.websocket_router.websocket("/ws")
    async def websocket_endpoint(ws):
        await ws.accept()
        await ws.close()

    middleware = types.ModuleType("middleware")

    class GlobalExceptionMiddleware:
        pass

    def setup_middleware(app: FastAPI) -> None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    middleware.setup_middleware = setup_middleware
    middleware.GlobalExceptionMiddleware = GlobalExceptionMiddleware

    sys.modules["conversation_service.api.routes"] = routes
    sys.modules["conversation_service.api.middleware"] = middleware
    builtins.run_core_validation = lambda: None

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield

    builtins.lifespan = _lifespan
    import conversation_service.main as cs_main
    importlib.reload(cs_main)
    return cs_main.create_app()


def test_create_app_registers_routes_and_middleware() -> None:
    app = build_app()

    routes = {route.path for route in app.routes}
    assert "/chat" in routes
    assert "/ws" in routes

    middlewares = [mw.cls for mw in app.user_middleware]
    assert CORSMiddleware in middlewares

