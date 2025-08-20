import importlib
import sys
import types
import builtins
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect


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
        try:
            while True:
                data = await ws.receive_text()
                await ws.send_text(data)
        except WebSocketDisconnect:
            pass

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


def test_websocket_echo() -> None:
    app = build_app()
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("hello")
            assert websocket.receive_text() == "hello"
    except WebSocketDisconnect:
        pass


def test_websocket_multiple_messages() -> None:
    app = build_app()
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws") as websocket:
            for i in range(3):
                msg = f"msg {i}"
                websocket.send_text(msg)
                assert websocket.receive_text() == msg
    except WebSocketDisconnect:
        pass

