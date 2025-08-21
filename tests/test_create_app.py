from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware


def build_app() -> FastAPI:
    app = FastAPI()
    router = APIRouter()

    @router.get("/chat")
    async def chat_endpoint():
        return {"ok": True}

    ws_router = APIRouter()

    @ws_router.websocket("/ws")
    async def websocket_endpoint(ws):
        await ws.accept()
        await ws.close()

    app.include_router(router)
    app.include_router(ws_router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def test_create_app_registers_routes_and_middleware() -> None:
    app = build_app()

    routes = {route.path for route in app.routes}
    assert "/chat" in routes
    assert "/ws" in routes

    middlewares = [mw.cls for mw in app.user_middleware]
    assert CORSMiddleware in middlewares

